import re
import os

def simplify_sql_views(input_filename="Views_Owner_BI.sql", output_filename="Views_Owner_BI_simplified.txt"):
    """
    Legge un file SQL contenente definizioni di viste e crea un file di testo
    semplificato con nomi delle viste e delle colonne principali.
    """
    simplified_schema = []
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            in_view_definition = False
            current_view_name = None
            current_columns = []
            in_select_clause = False

            for line in infile:
                line_strip = line.strip()

                # Rileva l'inizio di una nuova definizione di vista
                view_match = re.match(r"CREATE\s+VIEW\s+(\[bi\]\.\[\w+\])", line_strip, re.IGNORECASE)
                if view_match:
                    
                    if current_view_name and current_columns:
                         simplified_schema.append(f"VIEW: {current_view_name} COLUMNS: {', '.join(current_columns)}")

                    current_view_name = view_match.group(1)
                    current_columns = []
                    in_view_definition = True
                    in_select_clause = False
                   
                    continue 

                
                if in_view_definition:
                    # Rileva l'inizio della clausola SELECT
                    if re.match(r"SELECT", line_strip, re.IGNORECASE):
                        in_select_clause = True
                        cols_on_select_line = re.findall(r"(\[?\w+\]?)\s*(?:,|FROM|\s*$)", line_strip[6:].strip(), re.IGNORECASE)
                        for col in cols_on_select_line:
                             clean_col = col.replace('[', '').replace(']', '')
                             if clean_col and clean_col.upper() not in ['SELECT', 'AS']:
                                 current_columns.append(clean_col)
                        continue 

                    # Rileva la fine della sezione colonne (inizio FROM)
                    if re.match(r"FROM", line_strip, re.IGNORECASE):
                        in_select_clause = False
                        in_view_definition = False
                        
                        if current_view_name and current_columns:
                             simplified_schema.append(f"VIEW: {current_view_name} COLUMNS: {', '.join(current_columns)}")
                        current_view_name = None
                        current_columns = []
                        continue

                    
                    if in_select_clause:
                        
                        line_no_comments = line_strip.split('--')[0]
                        potential_cols = re.findall(r"(\[?\w+\]?)(?:\s*,?\s*$)", line_no_comments)

                        if potential_cols:
                            last_potential_col = potential_cols[-1]
                            clean_col = last_potential_col.replace('[', '').replace(']', '')
                            if clean_col and clean_col.upper() not in ['AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'CONVERT', 'ISNULL', 'COALESCE', 'LEFT', 'RIGHT', 'FORMAT', 'DATEADD', 'DATEDIFF', 'YEAR', 'MONTH', 'DAY', 'GETDATE', 'CAST', 'IIF']: # Ignora parole chiave SQL comuni
                                current_columns.append(clean_col)
                                


            
            if current_view_name and current_columns:
                 simplified_schema.append(f"VIEW: {current_view_name} COLUMNS: {', '.join(current_columns)}")

        # Scrivi l'output semplificato
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for line in simplified_schema:
                outfile.write(line + "\n")

        print(f"Schema semplificato scritto su: {output_filename}")
        print(f"Numero di viste processate: {len(simplified_schema)}")

    except FileNotFoundError:
        print(f"Errore: File di input '{input_filename}' non trovato.")
    except Exception as e:
        print(f"Errore durante l'elaborazione del file: {e}")

# --- Esegui la funzione ---
if __name__ == "__main__":
    simplify_sql_views()