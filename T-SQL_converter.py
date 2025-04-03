import json
import re
import os
import logging
from pathlib import Path
from datasets import load_dataset, load_from_disk

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_and_prepare_dataset(dataset_path="merged_dataset_optimized_pre_conv"):
    try:
        dataset = load_from_disk(dataset_path)
        logging.info(f'Dataset "{dataset_path}" loaded from disk.')
        return dataset
    except Exception as e:
        logging.error(f'Error loading dataset from disk "{dataset_path}": {e}')
        return None


def convert_schema_to_tsql(schema):
    if not schema or not isinstance(schema, str):
        return schema

    # Pattern per catturare la CREATE TABLE statement (attenzione: regex semplificata)
    create_table_pattern = re.compile(r'CREATE\s+TABLE\s+([`"\[]?\w+[`"\]]?)\s*\((.*?)\)', flags=re.IGNORECASE | re.DOTALL | re.UNICODE)

    def replace_create_table(match):
        table_name_raw = match.group(1)
        columns_def = match.group(2).strip()

        # Normalizza il nome della tabella a [nome]
        table_name = table_name_raw.strip('`"[]')
        table_name = f"[{table_name}]"

        # Split delle colonne: attenzione a non separare dentro eventuali parentesi
        columns_parts = re.split(r',(?![^()]*\))', columns_def)
        converted_columns = []
        for col_part in columns_parts:
            col_part = col_part.strip()
            if not col_part:
                continue

            # 1. Gestione PRIMARY KEY AUTOINCREMENT
            col_part = re.sub(
                r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
                r'INT IDENTITY(1,1) PRIMARY KEY',
                col_part,
                flags=re.IGNORECASE
            )
            col_part = re.sub(
                r'INTEGER\s+PRIMARY\s+KEY(?!.*IDENTITY)',
                r'INT PRIMARY KEY',
                col_part,
                flags=re.IGNORECASE
            )

            # 2. Conversione tipi di dati
            conversions = {
                r'\bINTEGER\b': r'INT',
                r'\bREAL\b': r'FLOAT',
                r'\bTEXT\b': r'NVARCHAR(MAX)',
                r'\bBLOB\b': r'VARBINARY(MAX)',
                r'\bBOOLEAN\b': r'BIT'
            }
            for pattern, repl in conversions.items():
                col_part = re.sub(pattern, repl, col_part, flags=re.IGNORECASE)

            # 3. Gestione VARCHAR/NVARCHAR senza specifica lunghezza
            col_part = re.sub(r'\bVARCHAR\b(?!\s*\()', r'VARCHAR(255)', col_part, flags=re.IGNORECASE)
            col_part = re.sub(r'\bNVARCHAR\b(?!\s*\()', r'NVARCHAR(255)', col_part, flags=re.IGNORECASE)

            # 4. Conversione degli identificatori delimitati (backticks o doppie virgolette)
            col_part = re.sub(r'[`"]([^`"]+)[`"]', r'[\1]', col_part)
            converted_columns.append(col_part)
        columns_str = ',\n  '.join(converted_columns)
        return f"CREATE TABLE {table_name} (\n  {columns_str}\n)"

    converted_schema = re.sub(create_table_pattern, replace_create_table, schema)
    return converted_schema


def convert_query_to_tsql(query):
    if not query or not isinstance(query, str):
        return query

    # Normalizzazione degli spazi bianchi
    query = re.sub(r'\s+', ' ', query).strip()

    # Protezione delle stringhe letterali
    single_quoted_strings = {}

    def single_quote_replacer(match):
        placeholder = f"__SINGLE_QUOTED_{len(single_quoted_strings)}__"
        single_quoted_strings[placeholder] = match.group(0)
        return placeholder

    query = re.sub(r"'([^']*)'", single_quote_replacer, query)

    # Conversione degli identificatori delimitati da backticks o doppie virgolette
    query = re.sub(r'[`"]([^`"]+)[`"]', r'[\1]', query)

    # Conversione delle stringhe tra doppi apici in apici singoli
    def double_quote_string_replacer(match):
        content = match.group(1).replace("'", "''")
        return f"'{content}'"
    query = re.sub(r'"([^"]*)"', double_quote_string_replacer, query)

    # Ripristino delle stringhe protette
    for placeholder, original in single_quoted_strings.items():
        query = query.replace(placeholder, original)

    # Gestione di LIMIT / OFFSET
    limit_offset_pattern = re.compile(r'\bLIMIT\s+(\d+)\s*(?:OFFSET\s+(\d+)\s*)?$', flags=re.IGNORECASE)
    limit_match = re.search(limit_offset_pattern, query)
    if limit_match:
        limit_val = limit_match.group(1)
        offset_val = limit_match.group(2) if limit_match.group(2) else "0"
        query = re.sub(limit_offset_pattern, '', query).strip()
        if offset_val == "0":
            query = re.sub(r'(SELECT(\s+DISTINCT)?)\s+', rf'\1 TOP {limit_val} ', query, flags=re.IGNORECASE, count=1)
        else:
            if not re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE):
                query += " ORDER BY (SELECT NULL)"
            query += f" OFFSET {offset_val} ROWS FETCH NEXT {limit_val} ROWS ONLY"

    # Sostituzioni di funzioni
    query = re.sub(r'\bSUBSTR\s*\(', 'SUBSTRING(', query, flags=re.IGNORECASE)
    query = re.sub(r"strftime\(\s*'%Y'\s*,\s*([^)]+)\)", r'YEAR(\1)', query, flags=re.IGNORECASE)
    query = re.sub(r"strftime\(\s*'%m'\s*,\s*([^)]+)\)", r'MONTH(\1)', query, flags=re.IGNORECASE)
    query = re.sub(r"strftime\(\s*'%d'\s*,\s*([^)]+)\)", r'DAY(\1)', query, flags=re.IGNORECASE)

    # Conversione dell'operatore di concatenazione da || a +
    operand_pattern = r"(\[[^\]]+\]|\w+|'[^']*')"
    while re.search(rf'{operand_pattern}\s*\|\|\s*{operand_pattern}', query, re.IGNORECASE):
        query = re.sub(rf'({operand_pattern})\s*\|\|\s*({operand_pattern})', r'\1 + \2', query, flags=re.IGNORECASE)


    # Conversione dei booleani
    query = re.sub(r'\bTRUE\b', '1', query, flags=re.IGNORECASE)
    query = re.sub(r'\bFALSE\b', '0', query, flags=re.IGNORECASE)

    # Sostituzione di IFNULL con ISNULL
    query = re.sub(r'\bIFNULL\s*\(', 'ISNULL(', query, flags=re.IGNORECASE)

    # Sostituzione di RANDOM() con RAND() e TOTAL() con SUM()
    query = re.sub(r'\bRANDOM\s*\(\)', 'RAND()', query, flags=re.IGNORECASE)
    query = re.sub(r'\bTOTAL\s*\(', 'SUM(', query, flags=re.IGNORECASE)

    # Gestione del tipo DATETIME
    query = re.sub(r'\bDATETIME\b', 'DATETIME2', query, flags=re.IGNORECASE)

    # Conversione di join impliciti (semplice, con limiti evidenti)
    join_pattern = re.compile(
        r'FROM\s+([^\s,]+)\s*,\s*([^\s,]+)\s+WHERE\s+([^\s\.]+)\.([^\s=]+)\s*=\s*([^\s\.]+)\.([^\s=]+)',
        flags=re.IGNORECASE
    )
    query = re.sub(join_pattern, r'FROM \1 INNER JOIN \2 ON \3.\4 = \5.\6 WHERE', query)

    # Correzioni per valori letterali
    def bracketed_numeric_literal_replacer(match):
        operator = match.group(1)
        numeric_value = match.group(2)
        return f"{operator} {numeric_value}"
    query = re.sub(r'([=<>!]+\s*)\[(\d+(?:\.\d+)?)\]', bracketed_numeric_literal_replacer, query)

    def bracketed_string_literal_replacer(match):
        operator = match.group(1)
        string_value = match.group(2).replace("'", "''")
        return f"{operator} '{string_value}'"
    query = re.sub(r'([=<>!]+\s*)\[([^\]]+)\]', bracketed_string_literal_replacer, query)

    # Rimozione di apici singoli per valori numerici
    numeric_string_pattern = r'([=<>!]+\s*)\'(\d+(?:\.\d+)?)\'(?!\s*\w)'
    query = re.sub(numeric_string_pattern, r'\1\2', query)

    # Aggiunta di CAST per confronti tra VARCHAR e numeri
    def add_cast_replacer(match):
        field = match.group(1)
        operator = match.group(2)
        number = match.group(3)
        return f"TRY_CAST({field} AS FLOAT) {operator} {number}"
    compare_pattern = r'(\[[^\]]+\])\s*([=<>!]+)\s*(\d+(?:\.\d+)?)\b'
    query = re.sub(compare_pattern, add_cast_replacer, query)

    return query.strip()


def process_dataset(dataset):
    if dataset is None:
        logging.error("Dataset non caricato, impossibile processare.")
        return None

    converted_dataset = {}
    schema_fields = ['schema', 'database_schema', 'db_schema', 'context']
    query_fields = ['answer', 'query']

    for split in dataset.keys():
        split_data = dataset[split]
        converted_examples = []
        logging.info(f"Processing split: {split} con {len(split_data)} esempi...")
        for count, example in enumerate(split_data, start=1):
            converted_example = example.copy()
            schema_converted = False

            # Conversione degli schemi
            for field in schema_fields:
                if field in converted_example and isinstance(converted_example[field], str):
                    original_content = converted_example[field]
                    converted_content = convert_schema_to_tsql(original_content)
                    if "CREATE TABLE" in original_content.upper():
                        converted_example[field] = converted_content
                        if field != 'context':
                            schema_converted = True

            # Conversione delle query
            for field in query_fields:
                if field in converted_example and isinstance(converted_example[field], str):
                    converted_example[field] = convert_query_to_tsql(converted_example[field])

            # Conversione dei blocchi di codice nel campo 'context'
            if 'context' in converted_example and isinstance(converted_example['context'], str) and not schema_converted:
                context = converted_example['context']
                # Blocchi SQL
                query_pattern = re.compile(r'```sql\s*(.*?)\s*```', flags=re.DOTALL | re.IGNORECASE)
                matches = re.findall(query_pattern, context)
                for match in matches:
                    converted_query_block = convert_query_to_tsql(match)
                    original_block = f"```sql{match}```"
                    new_block = f"```sql\n{converted_query_block}\n```"
                    context = context.replace(original_block, new_block, 1)

                # Blocchi schema
                schema_pattern = re.compile(r'```schema\s*(.*?)\s*```', flags=re.DOTALL | re.IGNORECASE)
                schema_matches = re.findall(schema_pattern, context)
                for match in schema_matches:
                    converted_schema_block = convert_schema_to_tsql(match)
                    original_block = f"```schema{match}```"
                    new_block = f"```schema\n{converted_schema_block}\n```"
                    context = context.replace(original_block, new_block, 1)

                converted_example['context'] = context

            converted_examples.append(converted_example)
            if count % 100 == 0:
                logging.info(f"  Processati {count}/{len(split_data)} esempi...")
        converted_dataset[split] = converted_examples
        logging.info(f"Elaborazione del split '{split}' completata.")

    return converted_dataset


def save_converted_dataset(converted_dataset, output_dir="t-sql_dataset_corrected"):
    if converted_dataset is None:
        logging.error("Nessun dataset convertito da salvare.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split, examples in converted_dataset.items():
        output_path = os.path.join(output_dir, f"{split}.json")
        logging.info(f"Salvataggio di {len(examples)} esempi per lo split '{split}' in {output_path}...")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Errore nel salvataggio del file {output_path}: {e}")
    logging.info(f"Dataset convertito salvato nella directory: {output_dir}")


def main():
    logging.info("Inizio conversione del dataset da SQLite a T-SQL...")
    dataset = download_and_prepare_dataset()
    converted_dataset = process_dataset(dataset)
    save_converted_dataset(converted_dataset)
    logging.info("Conversione completata!")


if __name__ == "__main__":
    main()
