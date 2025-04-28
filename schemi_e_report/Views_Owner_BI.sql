USE [DBA_IntergeaNE]
GO

/****** Object:  View [bi].[AUTO_aree]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[AUTO_aree]
as
select v.__ID as id_area, CodiceArea as codice_area, Descrarea descr_area, gv.Descrizione gruppo_area
from ANAG_Auto_area v
left join tree.Tree_Link_Auto_GruppoNaturaVendita lv on v.__ID=lv.IdDimensione
left join tree.TREE_Auto_GruppoNaturaVendita gv on gv.IDRamo=lv.IdRamo;
GO

/****** Object:  View [bi].[AUTO_aziende]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[AUTO_aziende]
as 
SELECT CodiceAzienda as id_azienda,
	   case CodiceAzienda when 'GF' then 'Gruppo Ferrari'
	   when 'A9' then 'Autoteam 9'
	   when 'AT' then 'Autoteam'
	   when 'CV' then 'Car Village' end as descr_azienda
FROM [DBA_IntergeaNE].[dbo].[ANAG_Azienda];
GO

/****** Object:  View [bi].[AUTO_canali_vendita]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[AUTO_canali_vendita]
as
select v.__ID as id_canale_vendita, CodiceCanaleVendita as codice_canale_vendita, DescrCanaleVendita as descr_canale_vendita, gv.Descrizione as gruppo_canale_vendita
from ANAG_Auto_canalevendita v
left join tree.Tree_Link_Auto_Gruppocanalevendita lv on v.__ID=lv.IdDimensione
left join tree.TREE_Auto_Gruppocanalevendita gv on gv.IDRamo=lv.IdRamo;
GO

/****** Object:  View [bi].[AUTO_clienti]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- bi.AUTO_dim_clienti source

CREATE view [bi].[AUTO_clienti]
as
select 
 c.__ID as id_cliente, c.CodiceCliente as codice_cliente, c.DescrizioneCliente as descr_cliente, c.CodiceFiscaleCliente as codice_fiscale_cliente, c.CAPCliente as cap_cliente
 , c.CodiceTipoCliente as codice_tipo_cliente,c.DescrizioneTipoCliente as tipo_cliente,  gtc.Descrizione as gruppo_tipo_cliente
 , codice_master_anagrafica as codice_master_cliente
from ANAG_Auto_ClienteAuto c
left join ANAG_Auto_TipoCliente tc
on c.CodiceTipoCliente = tc.CodiceTipoCliente
left join Tree.Tree_Link_Auto_GruppoTipoCliente ltc on ltc.IdDimensione=tc.__ID
left join tree.TREE_Auto_GruppoTipoCliente gtc on ltc.IdRamo=gtc.IDRamo;
GO

/****** Object:  View [bi].[AUTO_contatti_tipi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE view [bi].[AUTO_contatti_tipi]
as
select v.__ID as id_tipo_contatto, CodiceTipoContatto as codice_tipo_contatto, Descrtipocontatto as descr_tipo_contatto, gv.Descrizione gruppo_tipo_contatto
from ANAG_Auto_tipocontatto v
left join tree.Tree_Link_Auto_Gruppotipocontatto lv on v.__ID=lv.IdDimensione
left join tree.TREE_Auto_Gruppotipocontatto gv on gv.IDRamo=lv.IdRamo;
GO

/****** Object:  View [bi].[AUTO_contratti]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- bi.AUTO_contratti source

CREATE view [bi].[AUTO_contratti]
 as
 SELECT   
	  con_codiceAzienda+'_'+ [CON_IdContratto] as id_contratto
	  ,dim_Veicolo as id_veicolo
      ,[dim_Contratto_INF] as id_contratto_redd  	 
	  , dim_Area as id_area 
      ,format(con_annocontratto,'####-')+ con_codiceAzienda+'-'+ left(con_idcontratto,1)+format( [CON_NumeroContratto], ' #######') as nr_contratto  	  
	  ,dim_TipoContratto as id_tipo_contratto 
 	  ,dim_Provenienza as id_provenienza   
      ,dim_CanaleVendita as id_canale_vendita	 
	  ,dim_ClienteAuto as id_cliente
	  ,dim_tipocontatto as id_tipo_contatto
	  , nullif(CON_Segnalatore, '(Non definito)') as segnalatore_contratto 
	  , dim_Sede as id_sede
	  ,dim_Venditore as id_venditore
	  , convert(date,[CON_DataApertura] ) as dt_apertura_contratto 
      , convert(date,[CON_DataChiusura] ) as dt_chiusura_contratto 
      , CON_FlagAssicurazioni as has_assicurazione_contratto 
      , CON_FlagFinanziamento as has_finanziamento_contratto 
	  , isnull(CON_codGaranziaInterna, 'ND') as codice_garanzia_interna_contratto 
	  , replace(isnull( CON_desGaranziaInterna,'-'),'(Non definito)','-') as garanzia_interna_contratto 
      ,[CON_ImportoContratto] as imp_netto_contratto  
      ,[CON_ImportoFinanziamento] as imp_netto_finanziato_contratto 
      ,[CON_ImportoListino]	as imp_netto_listino_contratto 
      ,[CON_NumeroPermute] as count_permute_contratto 
      ,[CON_ValorePermuta] as imp_netto_permute_contratto 
	  , CON_CodiceStatoContratto as codice_status_contratto 
	  , CON_DescrStatoContratto	 as status_contratto 
	  , CON_id_preventivo as id_preventivo
  FROM [dbo].[FATTI_Auto_Contratti_INF] f
  union  
   SELECT
       con_codiceAzienda+'_'+ [CON_IdContratto] as dim_contratto
	   ,dim_Veicolo as dim_veicolo	
      ,[dim_Contratto_INF] as dim_contratto_redd  	  
	  , dim_Area as dim_area 
      ,format(con_annocontratto,'####-')+ con_codiceAzienda+'-'+ left(con_idcontratto,1)+format( [CON_NumeroContratto], ' #######')  as numero_contratto 	  
	  ,dim_TipoContratto as dim_tipo_contratto 
 	  ,dim_Provenienza as dim_provenienza     
      ,dim_CanaleVendita as dim_canale_vendita	 
	  ,dim_ClienteAuto as dim_cliente
	  
	  ,dim_tipocontatto as dim_tipo_contatto
	  , nullif(CON_Segnalatore, '(Non definito)') as segnalatore_contratto 
	  , dim_Sede as dim_sede
	  ,dim_Venditore as dim_venditore 
	  , convert(date,[CON_DataApertura] ) as dt_apertura_contratto
      , convert(date,[CON_DataChiusura] ) as dt_chiusura_contratto
      , CON_FlagAssicurazioni  as has_assicurazione_contratto 
      , CON_FlagFinanziamento as has_finanziamento_contratto 
	  , isnull(CON_codGaranziaInterna, 'ND')  as codice_garanzia_interna_contratto 
	  , replace(isnull( CON_desGaranziaInterna,'-'),'(Non definito)','-') as garanzia_interna_contratto 
      ,[CON_ImportoContratto] as imp_netto_contratto  
      ,[CON_ImportoFinanziamento] as imp_netto_finanziato_contratto
      ,[CON_ImportoListino]	as imp_netto_listino_contratto
      ,[CON_NumeroPermute] as count_permute_contratto 
      ,[CON_ValorePermuta] as imp_netto_permute_contratto
	  , CON_CodiceStatoContratto as codice_status_contratto
	  , CON_DescrStatoContratto	 as status_contratto
	  , con_id_preventivo as dim_preventivo
  FROM [dbo].[FATTI_Auto_Contratti_STO] f;
GO

/****** Object:  View [bi].[AUTO_contratti_redd]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- bi.AUTO_contratti_redd source

-- bi.AUTO_contratti_redd source

CREATE view [bi].[AUTO_contratti_redd]
as

select dim_Contratto_INF as id_contratto_redd,
cod_tipo_CR as codice_tipo_costo_ricavo_redd,
tipo_CR as descr_tipo_costo_ricavo_redd,
cod_costo as codice_costo_redd,
RED_descr_costo as descr_costo_redd,
cod_spesa as codice_spesa_redd,
RED_descr_spesa as descr_spesa_redd, 
des_totale_parziale as tipo_totale_redd,
des_margine as descr_margine_redd,
liv_margine as liv_margine_redd, 
costo_prev as imp_costo_previsto_redd,
costo_prev_venditore as imp_costo_previsto_venditore_redd,
costo_cons as imp_costo_consuntivo_redd,
null as imp_ricavo_previsto_redd,
null as imp_ricavo_previsto_venditore_redd, 
null as imp_ricavo_consuntivo_redd, 
null as imp_netto_provvigione_redd,
idx_cont_costo as idx_contabile_costo_redd,
idx_cont_spesa as idx_contabile_spesa_redd,
desc_riga as descr_riga_redd,
case when [cod_costo] is not null 
then [RED_codiceAzienda]+'_C_'+[cod_costo]
when [cod_spesa] is not null 
then [RED_codiceAzienda]+'_C_'+[cod_spesa]  
else [RED_codiceAzienda]+'_C_'+cod_tipo_CR  end codice_spesa_costo_redd
   FROM [dbo].[FATTI_Auto_Redd_INF]f
   left join Tree.TREE_Auto_GruppoProdottiAssicurativi a
   on f.ric_gruppoProdottiAssicurativi = a.IDRamo
   where isnull(costo_cons,0)+ISNULL(costo_prev,0)<>0

union all
--parte per la parte ricavi
select dim_contratto_INF as id_contratto_redd,
cod_tipo_CR as codice_tipo_costo_ricavo_redd,
tipo_CR as tipo_costo_ricavo_redd,
cod_costo as codice_costo_redd,
RED_descr_costo as descr_costo_redd,
cod_spesa as codice_spesa_redd,
RED_descr_spesa as descr_spesa_redd, 
des_totale_parziale as tipo_totale_redd,
des_margine as desr_margine_redd,
liv_margine as liv_margine_redd, 
null as imp_costo_previsto_redd,
null as imp_costo_previsto_venditore_redd,
null as imp_costo_consuntivo_redd,
ricavo_prev as imp_ricavo_previsto_redd,
ricavo_prev_venditore as imp_ricavo_previsto_venditore_redd, 
ricavo_cons as imp_ricavo_consuntivo_redd, 
importo_provvigione as imp_imp_netto_provvigione_redd,
idx_cont_costo as idx_contabile_costo_redd,
idx_cont_spesa as idx_contabile_spesa_redd,
desc_riga as descr_riga_redd,
case 
when [cod_spesa] is not null 
then [RED_codiceAzienda]+'_R_'+[cod_spesa] 
when [cod_costo] is not null 
then [RED_codiceAzienda]+'_R_'+[cod_costo]
else [RED_codiceAzienda]+'_R_'+cod_tipo_CR  end codice_spesa_costo_redd
   FROM [dbo].[FATTI_Auto_Redd_INF] f
      left join Tree.TREE_Auto_GruppoProdottiAssicurativi a
   on f.ric_gruppoProdottiAssicurativi = a.IDRamo
   where isnull(ricavo_cons,0)+ISNULL(ricavo_prev,0)<>0

------------
union----aggiungo lo storico
select dim_Contratto_INF as id_contratto_redd,
cod_tipo_CR as codice_tipo_costo_ricavo_redd,
tipo_CR as tipo_costo_ricavo_redd,
cod_costo as codice_costo_redd,
RED_descr_costo as descr_costo_redd,
cod_spesa as codice_spesa_redd,
RED_descr_spesa as descr_spesa_redd, 
des_totale_parziale as tipo_totale_redd,
des_margine as desr_margine_redd,
liv_margine as liv_margine_redd, 
costo_prev as imp_costo_previsto_redd,
costo_prev_venditore as imp_costo_previsto_venditore_redd,
costo_cons as imp_costo_consuntivo_redd,
null as imp_ricavo_previsto_redd,
null as imp_ricavo_previsto_venditore_redd, 
null as imp_ricavo_consuntivo_redd, 
null as imp_imp_netto_provvigione_redd,
idx_cont_costo as idx_contabile_costo_redd,
idx_cont_spesa as idx_contabile_spesa_redd,
desc_riga as descr_riga_redd,
case when [cod_costo] is not null 
then [RED_codiceAzienda]+'_C_'+[cod_costo]
when [cod_spesa] is not null 
then [RED_codiceAzienda]+'_C_'+[cod_spesa]  ----ho il costo caricato sul codice spesa (è un fittizio) quindi creo una voce fittizia di costo
else [RED_codiceAzienda]+'_C_'+cod_tipo_CR  end codice_spesa_costo_redd
   FROM [dbo].[FATTI_Auto_Redd_STO]f
   left join Tree.TREE_Auto_GruppoProdottiAssicurativi a
   on f.ric_gruppoProdottiAssicurativi = a.IDRamo
   where isnull(costo_cons,0)+ISNULL(costo_prev,0)<>0

union all
--parte per la parte ricavi
select dim_contratto_INF as id_contratto_redd,
cod_tipo_CR as codice_tipo_costo_ricavo_redd,
tipo_CR as tipo_costo_ricavo_redd,
cod_costo as codice_costo_redd,
RED_descr_costo as descr_costo_redd,
cod_spesa as codice_spesa_redd,
RED_descr_spesa as descr_spesa_redd, 
des_totale_parziale as tipo_totale_redd,
des_margine as desr_margine_redd,
liv_margine as liv_margine_redd, 
null  as imp_costo_previsto_redd,
null  as imp_costo_previsto_venditore_redd,
null  as imp_costo_consuntivo_redd,
ricavo_prev as imp_ricavo_previsto_redd,
ricavo_prev_venditore as imp_ricavo_previsto_venditore_redd, 
ricavo_cons as imp_ricavo_consuntivo_redd, 
importo_provvigione as imp_netto_provvigione_redd,
idx_cont_costo as idx_contabile_costo_redd,
idx_cont_spesa as idx_contabile_spesa_redd,
desc_riga as descr_riga_redd,
case 
when [cod_spesa] is not null 
then [RED_codiceAzienda]+'_R_'+[cod_spesa] 
when [cod_costo] is not null 
then [RED_codiceAzienda]+'_R_'+[cod_costo]
else [RED_codiceAzienda]+'_R_'+cod_tipo_CR  end codice_spesa_costo_redd
   FROM [dbo].[FATTI_Auto_Redd_STO] f
      left join Tree.TREE_Auto_GruppoProdottiAssicurativi a
   on f.ric_gruppoProdottiAssicurativi = a.IDRamo
   where isnull(ricavo_cons,0)+ISNULL(ricavo_prev,0)<>0;
GO

/****** Object:  View [bi].[AUTO_contratti_tipi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE view [bi].[AUTO_contratti_tipi]
as
select v.__ID as id_tipo_contratto, CodiceTipoContratto as codice_tipo_contratto, DescrTipoContratto as descr_tipo_contratto, gv.Descrizione gruppo_tipo_contratto
from ANAG_Auto_TipoContratto v
left join tree.Tree_Link_Auto_GruppoTipoContratto lv on v.__ID=lv.IdDimensione
left join tree.TREE_Auto_GruppoTipoContratto gv on gv.IDRamo=lv.IdRamo;
GO

/****** Object:  View [bi].[AUTO_Preventivi_dopo marzo25]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




CREATE view [bi].[AUTO_Preventivi_dopo marzo25]
as
select PRV_CodiceAzienda codice_Azienda
,PRV_Id_preventivo Id_preventivo
,PRV_numeroPreventivo numero_Preventivo
, PRV_data_preventivo dt_Preventivo
, dim_Area
, dim_ClienteAuto
, PRV_ClienteContatto Cliente_Contatto
, dim_TipoVeicolo
, dim_Marca
, dim_Modello
, dim_MarcaModello
, dim_veicolo
, dim_Sede
, dim_Venditore
, PRV_ordine_fabbrica ordine_fabbrica
, PRV_numeroOrdineCommerciale numero_ordine_commerciale
, PRV_id_trattativa id_trattativa
, PRV_codiceTrattativa codice_trattativa
, PRV_status status_trattativa
, PRV_causaleAnnullamento causale_annullamento
, PRV_causaVisita causa_visita
, PRV_descrizioneVeicolo descr_veicolo
, PRV_targa	targa
, PRV_telaio telaio
, PRV_telCellulare tel_cellulare
, PRV_email email
, PRV_cliIndirizzo localita
, PRV_percentuale_chiusura_prev percentuale_chiusura_prev
, PRV_importo importo
, PRV_test_drive test_drive
, PRV_noteVenditore note_venditore
from dbo.FATTI_Auto_Preventivi_INF f
union 
select PRV_CodiceAzienda codice_Azienda
,PRV_Id_preventivo Id_preventivo
,PRV_numeroPreventivo numero_Preventivo
, PRV_data_preventivo dt_Preventivo
, dim_Area
, dim_ClienteAuto
, PRV_ClienteContatto Cliente_Contatto
, dim_TipoVeicolo
, dim_Marca
, dim_Modello
, dim_MarcaModello
, dim_veicolo
, dim_Sede
, dim_Venditore
, PRV_ordine_fabbrica ordine_fabbrica
, PRV_numeroOrdineCommerciale numero_ordine_commerciale
, PRV_id_trattativa id_trattativa
, PRV_codiceTrattativa codice_trattativa
, PRV_status status_trattativa
, PRV_causaleAnnullamento causale_annullamento
, PRV_causaVisita causa_visita
, PRV_descrizioneVeicolo descr_veicolo
, PRV_targa	targa
, PRV_telaio telaio
, PRV_telCellulare tel_cellulare
, PRV_email email
, PRV_cliIndirizzo localita
, PRV_percentuale_chiusura_prev percentuale_chiusura_prev
, PRV_importo importo
, PRV_test_drive test_drive
, PRV_noteVenditore note_venditore
from dbo.FATTI_Auto_Preventivi_STO f

GO

/****** Object:  View [bi].[AUTO_provenienze]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE view [bi].[AUTO_provenienze]
as
select v.__ID as id_provenienza, CodiceProvenienza as codice_provenienza, Descrprovenienza as descr_provenienza, gv.Descrizione as gruppo_provenienza
from ANAG_Auto_provenienza v
left join tree.Tree_Link_Auto_Gruppoprovenienza lv on v.__ID=lv.IdDimensione
left join tree.TREE_Auto_Gruppoprovenienza gv on gv.IDRamo=lv.IdRamo;
GO

/****** Object:  View [bi].[AUTO_sedi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[AUTO_sedi]
as
select sede.__ID as id_sede,
	   coalesce (case when tsede.tipo = 'SD' then tsede.descrizione 
				       when tsede_l2.tipo = 'SD' then tsede_l2.descrizione
					  else null end
				, Sede.descrsede_db, DescrSede)	descr_sede,
		coalesce (case when tsede.tipo = 'PV' then tsede.descrizione else null end
						,Sede.DescrPuntoVendita, DescrSede)	descr_punto_vendita

	  ,coalesce( case when tsede.tipo = 'AZ' then tsede.descrizione 
					  when tsede_l2.tipo = 'AZ' then tsede_l2.descrizione
					  when tsede_l3.tipo = 'AZ' then tsede_l3.descrizione
					else null end
			 , tsede_l3.descrizione, tsede_l2.descrizione,tsede.descrizione,sede.descrsede) gruppo_sede
--select *
  from  dbo.ANAG_Auto_Sede sede 
  left join tree.Tree_Link_Auto_GruppoSede l on sede.__ID = l.IdDimensione
  left join tree.TREE_Auto_GruppoSede		tsede on l.IdRamo=tsede.idramo
  left join tree.TREE_Auto_GruppoSede		tsede_l2 on tsede.IdParent=tsede_l2.idramo
  left join tree.TREE_Auto_GruppoSede		tsede_l3 on tsede_l2.IdParent=tsede_l3.idramo;
GO

/****** Object:  View [bi].[AUTO_veicoli]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE view [bi].[AUTO_veicoli] as
select
  v.[dim_veicolo] as id_veicolo,
  codice_azienda as id_azienda,
  [IdVeicolo] as id_gesionale_veicolo,
  v.[telaio] as telaio_veicolo,
  v.[targa] as targa_veicolo,
  fv.data_fattura_a as dt_acquisto_veicolo,
  v.[data_immatricolazione] as dt_immatricolazione_veicolo,
  data_arrivo as dt_arrivo_veicolo,
  data_ScadenzaBollo as dt_scadenza_bollo,
  data_consegna as dt_uscita_veicolo,
  coalesce(v.[Marca], m.descrmarchio) marca_veicolo,
  gm.Descrizione gruppo_marca_veicolo,
  coalesce(v.[Modello], mm.descrmodello) modello_veicolo,
  versione as versione_veicolo,
  [cod_Veicolo] as codice_interno_veicolo,
  [Veicolo] as veicolo,
  [tipo_veicolo],
  tv.DescrTipoVeicolo gruppo_tipo_veicolo,
  v.[alimentazione] as alimentazione_veicolo,
  ga.Descrizione gruppo_alimentazione_veicolo,
  case when Area='N' then 'Nuovo' else 'Usato' end as nuovo_usato_veicolo,
  destinazione as destinazione_al_ritiro_veicolo,
  [DestinazioneAllaVendita] as destinazione_alla_vendita_veicolo,
  [CanaleUscita] as canale_uscita_veicolo,
  [AreaCommerciale] as area_commerciale_veicolo,
  [km_percorsi] as km_percorsi_veicolo,
  [Linea] as linea_veicolo,
  fv.CodiceProvenienza as provenienza_veicolo,
  fv.[status_veicolo_desc] as status_veicolo,
  fv.CodiceTipoRitiro as codice_tipo_ritiro_veicolo,
  [UbicazioneAttuale] as ubicazione_attuale_veicolo,
  [UbicazioneRitiro] as ubicazione_al_ritiro_veicolo,
  [id_contratto_ritiro] as id_contratto_ritiro_veicolo,
  CodiceFornitore as codice_fornitore_veicolo,
  [NoteVeicolo] as note_veicolo
from
  ANAG_Auto_Veicolo_info v
  left join dbo.ANAG_TipoAlimentazione a on v.cod_alimentazione = a.CodiceTipoAlimentazione
  left join tree.Tree_Link_GruppoAlimentazione lga on lga.IdDimensione = a.__ID
  left join tree.TREE_GruppoAlimentazione ga on ga.IDRamo = lga.IdRamo
  left join dbo.ANAG_Auto_MarcaModello mm on v.cod_marca = mm.CodiceMarchio
  and v.cod_modello = mm.CodiceModello
  left join tree.Tree_Link_Auto_GruppoMarcaModello lgmm on lgmm.IdDimensione = mm.__ID
  left join tree.TREE_Auto_GruppoMarcaModello gmm on gmm.IDRamo = lgmm.IdRamo
  left join dbo.ANAG_Auto_Marca m on v.cod_marca = m.CodiceMarchio
  left join tree.Tree_Link_Auto_GruppoMarca lgm on lgm.IdDimensione = m.__ID
  left join tree.TREE_Auto_GruppoMarca gm on gm.IDRamo = lgm.IdRamo
  left join dbo.ANAG_TipoVeicolo tv on v.cod_tipo_veicolo = tv.CodiceTipoVeicolo
  left join tree.Tree_Link_GruppoTipoVeicolo ltv on ltv.IdDimensione = tv.__ID
  left join tree.TREE_GruppoAlimentazione gtv on gtv.IDRamo = ltv.IdRamo
  left join FATTI_Auto_Veicoli_INF fv on v.dim_veicolo = fv.dim_veicolo;
GO

/****** Object:  View [bi].[AUTO_venditori]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[AUTO_venditori]
as
select v.__ID as id_venditore, CodiceVenditore codice_venditore, DescrVenditore descr_venditore, gv.Descrizione gruppo_venditori
from ANAG_Auto_Venditore v
left join tree.Tree_Link_Auto_GruppoVenditori lv on v.__ID=lv.IdDimensione
left join tree.TREE_Auto_GruppoVenditori gv on gv.IDRamo=lv.IdRamo;
GO

/****** Object:  View [bi].[CONTT_contatti]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- bi.CONTT_contatti source

-- bi.CONTT_contatti source

CREATE view [bi].[CONTT_contatti] as
select all 
  [Cod_Azienda] as id_azienda,
  [CodiceClienteDWH] as [id_contatto]
, [ragione_sociale] as ragione_sociale_contatto
, [codice_fiscale] as codice_fiscale_contatto
, [par_iva] as partita_iva_contatto
, [indirizzo]  as indirizzo_contatto
, convert(varchar,[cap]) as cap_contatto
, [località] as localita_contatto
, provincia as provincia_contatto
, [nazione] as nazione_contatto
, [email1] as email_contatto
, [tel0] as telefono_contatto
, [tipo_soc] as tipo_soggetto_contatto
, [sesso] as sesso_contatto
, [cognome] as cognome_contatto
, [nome] as nome_contatto
, [codice_master_anagrafica] as codice_master_contatto
, [id_anagrafica] as id_anagrafica_contatto
, DBO.my_GREATEST_DATE(data_modifica_consenso, coalesce([data_modifica], [data_inserimento])) dt_modifica_contatto 
from IN_Clienti_AT
union 
select all 
  [Cod_Azienda] as id_azienda,
  [CodiceClienteDWH] as [id_contatto]
, [ragione_sociale] as ragione_sociale_contatto
, [codice_fiscale] as codice_fiscale_contatto
, [par_iva] as partita_iva_contatto
, [indirizzo]  as indirizzo_contatto
, convert(varchar,[cap]) as cap_contatto
, [località] as localita_contatto
, provincia as provincia_contatto
, [nazione] as nazione_contatto
, [email1] as email_contatto
, [tel0] as telefono_contatto
, [tipo_soc] as tipo_soggetto_contatto
, [sesso] as sesso_contatto
, [cognome] as cognome_contatto
, [nome] as nome_contatto
, [codice_master_anagrafica] as codice_master_contatto
, [id_anagrafica] as id_anagrafica_contatto
, DBO.my_GREATEST_DATE(data_modifica_consenso, coalesce([data_modifica], [data_inserimento])) dt_modifica_contatto 
from IN_Clienti_A9
union 
select all 
   [Cod_Azienda] as id_azienda,
   [CodiceClienteDWH] as [id_contatto]
, [ragione_sociale] as ragione_sociale_contatto
, [codice_fiscale] as codice_fiscale_contatto
, [par_iva] as partita_iva_contatto
, [indirizzo]  as indirizzo_contatto
, convert(varchar,[cap]) as cap_contatto
, [località] as localita_contatto
, provincia as provincia_contatto
, [nazione] as nazione_contatto
, [email1] as email_contatto
, [tel0] as telefono_contatto
, [tipo_soc] as tipo_soggetto_contatto
, [sesso] as sesso_contatto
, [cognome] as cognome_contatto
, [nome] as nome_contatto
, [codice_master_anagrafica] as codice_master_contatto
, [id_anagrafica] as id_anagrafica_contatto
, DBO.my_GREATEST_DATE(data_modifica_consenso, coalesce([data_modifica], [data_inserimento])) dt_modifica_contatto 
from IN_Clienti_CV
union 
select all 
[cod_Azienda] as id_azienda,
    [CodiceClienteDWH] as [id_contatto]
, [ragione_sociale] as ragione_sociale_contatto
, [codice_fiscale] as codice_fiscale_contatto
, [par_iva] as partita_iva_contatto
, [indirizzo]  as indirizzo_contatto
, convert(varchar,[cap]) as cap_contatto
, [località] as localita_contatto
, provincia as provincia_contatto
, [nazione] as nazione_contatto
, [email1] as email_contatto
, [tel0] as telefono_contatto
, [tipo_soc] as tipo_soggetto_contatto
, [sesso] as sesso_contatto
, [cognome] as cognome_contatto
, [nome] as nome_contatto
, [codice_master_anagrafica] as codice_master_contatto
, [id_anagrafica] as id_anagrafica_contatto
, DBO.my_GREATEST_DATE(data_modifica_consenso, coalesce([data_modifica], [data_inserimento])) dt_modifica_contatto 
from IN_Clienti_GF;
GO

/****** Object:  View [bi].[OFF_accettatori]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[OFF_accettatori] as 
SELECT [__ID] as id_accettatore, TES_DescrAccettatore as descr_accettatore
FROM DBA_IntergeaNE.dbo.ANAG_Officina_Accettatore;
GO

/****** Object:  View [bi].[OFF_clienti]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_clienti] as
select  o.__ID as id_cliente_officina
, CodiceCliFor as codice_cliente_officina
, DescrizioneCliFor as descr_cliente_officina
, tro.Descrizione as gruppo_cliente_officina
 from ANAG_Officina_ClienteOfficina o
  left join Tree.Tree_Link_Officina_GruppoClienteOfficina lo on lo.IdDimensione=o.__ID
  left join tree.TREE_Officina_GruppoClienteOfficina tro on tro.IDRamo=lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_commesse]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[OFF_commesse]
as
select 
	[dim_Commessa] as id_commessa,	
	t.[TES_CodiceAzienda] as id_azienda,
	[dim_TipoCommessa] as id_tipo_commessa,
	[dim_Officina] as id_officina,
	[dim_Cliente] as id_cliente_officina,
	[TES_IdCommessa] as id_gestionale_commessa,
	[TES_NumeroCommessa] as nr_commessa,
	[TES_DataApertura] as dt_apertura_commessa,
	[TES_DataChiusura] as dt_chiusura_commessa,
	tes_data_prevcons as dt_previsione_consegna_commessa,
	TES_data_iniziolavori as dt_inizio_lavoro_commessa,
	TES_data_finelavori as dt_fine_lavoro_commessa,
	TES_data_creazione as dt_creazione_commessa,
	TES_data_effettiva_consegna as dt_effettiva_consegna_commessa,
	TES_data_veicolo_pronto as dt_veicolo_pronto_commessa,
	[TES_CodiceMacroModello] as codice_macro_modello_commessa,
	[TES_DescrMacroModello] as descr_macro_modello_commessa,
	t.[TES_Telaio] as telaio_commessa,
	vei.__ID as id_veicolo_officina,
	TES_DescrAccettatore as accettatore_commessa,
	TES_PIVA as partita_iva_cliente_commessa,
	TES_segnoRiga as segno_riga_commessa,
	TES_mat_consumo as imp_netto_materiale_consumo_commessa,
	TES_spese_rifiuti as imp_netto_spese_rifiuti_commessa,
	TES_ODL as codice_odl_commessa,
	TES_EstGar as codice_estgar_commessa, 
	TES_ServCard as codice_servicard_commessa
 from [dbo].[FATTI_Officina_Testate_INF] t
  left join dbo.ANAG_Officina_Officina o on t.dim_Officina = o.__ID
  left outer join DBA_IntergeaNE.dbo.ANAG_Officina_Veicolo vei on vei.TES_NumeroVeicolo = t.TES_NumeroVeicolo
  WHERE ( TES_DataChiusura>='20210701' or TES_DataChiusura is null);


GO

/****** Object:  View [bi].[OFF_commesse_inconvenienti]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_commesse_inconvenienti]
as
select   
	 i.INC_codAzienda + CONVERT(varchar, i.INC_idCommessa)+'_'+convert(varchar,i.INC_idInconveniente) id_inconveniente
	,[dim_Commessa] as id_commessa
	, INC_prg_inconv as id_gestionale_inconveniente
	, dim_TipoIntervento as id_tipo_intervento
	, i.INC_TipoRiga as tipo_riga_inconveniente
	, convert(varchar,i.INC_numInconveniente) as riga_inconveniente
	, ISNULL(i.INC_codInconveniente,'-') as codice_inconveniente
	, isnull(i.INC_desCodInconveniente,'-') as descr_breve_inconveniente
	,aoi.INC_desInconveniente as descr_inconveniente
	, i.INC_tipoInconveniente as tipo_inconveniente
	, i.INC_codiceCarico as codice_carico_inconveniente
	, i.tipoCaricoCliente as tipo_carico_cliente_inconveniente
	, i.tipoCaricoGaranzia as tipo_carico_garanzia_inconveniente
	, gi.Descrizione as gruppo_inconveniente
	from FATTI_Officina_Inconvenienti_INF i
	left outer join ANAG_Officina_Inconveniente aoi on aoi.INC_codAzienda + convert(varchar,aoi.INC_idCommessa) + '_' + convert(varchar,aoi.INC_prg_inconv)=i.INC_codAzienda + CONVERT(varchar, i.INC_idCommessa)+'_'+convert(varchar,i.INC_idInconveniente)
	left join Tree.TREE_Officina_GruppoInconveniente gi on i.ric_GruppoInconveniente=gi.IDRamo
	  left join Tree.Tree_Link_Officina_GruppoInconveniente lo on lo.IdDimensione=aoi.__ID
	where  CONVERT(varchar, i.INC_idCommessa)+'_'+convert(varchar,i.INC_idInconveniente) is not null;
GO

/****** Object:  View [bi].[OFF_commesse_lavorazioni]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_commesse_lavorazioni] as
select
	 isnull(o.dim_officina, t.dim_officina) id_officina
	, isnull(t.dim_tipocommessa,dim_tipocommessa.__ID) id_tipo_commessa
	, o.dim_commessa as id_commessa
	, t.dim_Accettatore id_accettatore
	, codazienda+ CONVERT(varchar, IdCommessa)+'_'+convert(varchar,idInconveniente) id_inconveniente
	, [desRigaLavorazione] descr_riga_lavorazione
	, [ProgressivoRiga] progressivo_riga_lavorazione
	, isnull([dim_Esecutore],0) id_esecutore
	, isnull([dim_TipoIntervento],0)[id_tipo_intervento] 
	, isnull(dim_tipoesecutore,40) id_tipoesecutore
   , um as um_lavorazione
   , [OreLavorate] ore_lavorazione
   , case when isnull([CostoOrario],0)=0	
				then 30*OreLavorate
			else [CostoOperatore] end costo_lavorazione
   , coalesce([DataMovimento], TES_DataChiusura, tes_dataapertura) dt_movimento_lavorazione
   , [codVoceLavorazione] codice_voce_lavorazione_lavorazione
   ,  coalesce(nullif(ric_GruppoVoce, 0),
		case when um ='H' then 12 --mo
			 when ric_gruppotipoesecutore = 2 then 13  ---lavori esterni
			 else 12
			 end)		gruppo_voce_lavorazione
	,dim_Voce as id_voce
   ,[OreFatturate] ore_fatturate_lavorazione
   , [quota_fatturato] as quota_fatturato_lavorazione
   ,isnull(nullif( [OreTempario],0),[oreLavorate]) ore_tempario_lavorazione
   , [CostoLavExt] costo_lavoro_ext_lavorazione
   , CostoOrario costo_orario_lavorazione
   

from FATTI_Officina_manodopera_INF o
left join FATTI_Officina_Testate_inf t
	on t.dim_Commessa=o.dim_Commessa
left join ANAG_Officina_TipoCommessa dim_tipocommessa on dim_tipocommessa.TES_codAzienda = o.CodAzienda and dim_TipoCommessa.TES_CodiceTipoCommessa = o.CodTipoDoc
  WHERE ( TES_DataChiusura>='20210701' or TES_DataChiusura is null)
GO

/****** Object:  View [bi].[OFF_commesse_righe]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- bi.OFF_commesse_righe source

-- bi.OFF_commesse_righe source

CREATE view [bi].[OFF_commesse_righe]
as
select
  r.[dim_Commessa] as id_commessa
 , [ORC_dataLavorazione] dt_lavorazione_commessa
  , convert(varchar,[ORC_numInconveniente]) nr_inconveniente_commessa
 , [ORC_codAzienda]+ convert(varchar,ORC_idCommessa )+ '_' + convert(varchar,ORC_idInconveniente) id_inconveniente
 , [ORC_numRiga] nr_riga_commessa
 ,[ORC_indTipoRiga] indicatore_tipo_riga_commessa
 , [ORC_codTipoCommessa] codice_tipo_commessa
 , ORC_codTipoArticolo precodice_articolo_commessa
 , ORC_codArticolo codice_articolo_commessa
 , ORC_codLavorazione codice_lavorazione_commessa
 , ORC_codiceConto codice_conto_commessa
 , tpesec.DescrTipoEsecutore as descr_tipo_esecutore_commessa
 , tro.descrizione as gruppo_tipo_esecutore_commessa
 , dim_Esecutore as id_esecutore
 ,r.dim_TipoIntervento as id_tipo_intervento
 , coalesce(nullif(ric_GruppoVoce, 0),  --se c'è prendo il raggruppamento del DBA (ma va per eccezione), altrimenti
		case when i.INC_codInconveniente like '%NOL' then 1-- vetture sostitutive
		     when ric_gruppoTipoEsecutore = 4 then 5--ricambi
			 when ric_gruppotipoesecutore = 2 then 13  ---lavori esterni
			 when ORC_codUmis ='H' then 12 --mo
			 else 10 ---altre voci del fatturato altro
			 end) gruppo_voce_commessa
 , dim_Voce as id_voce
 , ORC_costo imp_netto_costo_commessa
 , ORC_qta * ORC_costoMedio *  TES_segnoRiga * (isnull(ORC_segnoRiga,1)+1)/2  imp_netto_costo_medio --se il segno è meno annullo il costo
 , ORC_qta * ORC_costoCMP *  TES_segnoRiga * (isnull(ORC_segnoRiga,1)+1)/2  imp_netto_costo_medio_ponderato_commessa
 , isnull(ORC_flOmaggio,0) is_omaggio_commessa
 , ORC_costoLavoriTerzi * TES_segnoRiga imp_netto_costo_lavori_terzi_commessa
 , ORC_prezzoListino imp_netto_prezzo_listino_commessa
 , ORC_prezzoListino * isnull(ORC_tempoFatt ,ORC_Qta)* TES_segnoRiga  imp_netto_riga_listino_commessa
 , ORC_prezzoUnitario imp_netto_prezzo_unitario_commessa
 , ORC_Qta qta_commessa
 , isnull(ORC_imponibileNetto, ORC_prezzounitario * isnull(ORC_qta,1)) *  TES_segnoRiga  imp_netto_riga_commessa
 , isnull(ORC_segnoRiga, TES_segnoRiga) segno_riga_commessa
  , ORC_tempoFatt tempo_fatt_commessa
 , ORC_qta_tempario qta_tempario_commessa
 , case when isnull(gv.IdParent,0)<>18 --non è 18 =fatturato altro
		 and ric_gruppoTipoEsecutore = 2 --LAVORI TERZI  
		then ORC_Qta 
		else null end qta_fatt_ric_commessa
 , ORC_tipoOperazione as tipo_operazione_commessa
 , ORC_codUmis as codice_umis_commessa
 , ORC_percAddebitoCliente as perc_addebito_cliente_commessa
 , r.tipoCaricoGaranzia as tipo_carico_garanzia_commessa
 , r.tipoCaricoCliente as tipo_carico_cliente_commessa
 
 from [dbo].[FATTI_Officina_RigheCommessa_INF] r
 left join FATTI_Officina_Testate_INF t
 on r.dim_Azienda = t.dim_Azienda
 and r.ORC_idCommessa = t.TES_IdCommessa
 left join FATTI_Officina_Inconvenienti_INF i
 on r.dim_Azienda = i.dim_Azienda
 and r.ORC_idCommessa = i.INC_idCommessa
 and r.ORC_idInconveniente = i.INC_idInconveniente
 left join Tree.TREE_Officina_GruppoVoce gv
 on coalesce(nullif(ric_GruppoVoce, 0),  --se c'è prendo il raggruppamento del DBA (ma va per eccezione), altrimenti
		case when i.INC_codInconveniente like '%NOL' then 1-- vetture sostitutive
		     when ric_gruppoTipoEsecutore = 4 then 5--ricambi
			 when ric_gruppotipoesecutore = 2 then 13  ---lavori esterni
			 when ORC_codUmis ='H' then 12 --mo
			 else 10 ---altre voci del fatturato altro
			 end)= gv.IDRamo

left outer join ANAG_Officina_TipoEsecutore tpesec on tpesec.__ID=dim_TipoEsecutore
  left join Tree.Tree_Link_Officina_GruppoTipoEsecutore lo on lo.IdDimensione=tpesec.__ID
  left join tree.TREE_Officina_GruppoTipoEsecutore tro on tro.IDRamo=lo.IdRamo			 
 where ( TES_DataChiusura>='20210701' or TES_DataChiusura is null);
GO

/****** Object:  View [bi].[OFF_commesse_tipi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- bi.OFF_commesse_tipi source

CREATE view [bi].[OFF_commesse_tipi] as
select  o.__ID as id_tipo_commessa
,TES_CodiceTipoCommessa codice_tipo_commessa
, TES_DescrTipoCommessa descr_tipo_commessa
, tro.Descrizione gruppo_tipo_commessa
 from ANAG_Officina_TipoCommessa o
  left join Tree.Tree_Link_Officina_GruppoTipoCommessa lo on lo.IdDimensione=o.__ID
  left join tree.TREE_Officina_GruppoTipoCommessa tro on tro.IDRamo=lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_esecutori]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- bi.OFF_esecutori source

CREATE view [bi].[OFF_esecutori] as
  select o.__ID as id_esecutore
  , CodiceEsecutore codice_esecutore
  , DescrEsecutore descr_esecutore
  , Qualifica qualifica_esecutore
, tro.Descrizione gruppo_esecutore
  from  ANAG_Officina_Esecutore o
  left join Tree.Tree_Link_Officina_GruppoEsecutori lo on lo.IdDimensione=o.__ID
  left join tree.TREE_Officina_GruppoEsecutori tro on tro.IDRamo=lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_esecutori_tipi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_esecutori_tipi] as
  select o.__ID as id_tipo_esecutore
, CodiceTipoEsecutore as codice_tipo_esecutore
, DescrTipoEsecutore as descr_tipo_esecutore
, tro.Descrizione as gruppo_tipo_esecutore
  from  ANAG_Officina_TipoEsecutore o
  left join Tree.Tree_Link_Officina_GruppoTipoEsecutore lo on lo.IdDimensione=o.__ID
  left join tree.TREE_Officina_GruppoTipoEsecutore tro on tro.IDRamo=lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_interventi_tipi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE view [bi].[OFF_interventi_tipi] as
  select o.__ID as id_tipo_intervento 
  ,CodiceTipoIntervento as codice_tipo_intervento
  ,DescrTipoIntervento as descr_tipo_intervento
  ,TipoCarico  as tipo_carico_tipo_intervento
  ,TipoGaranzia  as tipo_garanzia_tipo_intervento
  ,TipoCliente  as tipo_cliente_tipo_intervento
from  ANAG_Officina_TipoIntervento o
  left join Tree.Tree_Link_Officina_Gruppotipointervento lo on lo.IdDimensione=o.__ID
  left join tree.TREE_Officina_Gruppotipointervento tro on tro.IDRamo=lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_noleggi]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- bi.OFF_noleggi source

-- bi.OFF_noleggi source

CREATE view [bi].[OFF_noleggi] as
select 
 [NOL_id_movimento] as id_riga_noleggio
, [codAzienda] as id_azienda
, [cd_cliente_fatt_nol] as id_contatto
, [NOL_targa] as targa_noleggio
, [NOL_telaio] as telaio_noleggio
, [NOL_modello_alternativo] as modello_alternativo_noleggio
, [NOL_cod_segmento] as codice_segmento_noleggio
, [NOL_linea] as linea_noleggio
, [NOL_data_ritiro] as dt_ritiro_noleggio
, [NOL_ora_ritiro] as ora_ritiro_noleggio
, [NOL_note_ritiro] as note_ritiro_noleggio
, [NOL_data_consegna] as dt_consegna_noleggio
, [NOL_ora_consegna] as ora_consegna_noleggio
, [NOL_note_consegna] as note_consegna_noleggio
, [NOL_km_ritiro] as km_ritiro_noleggio
, [NOL_km_consegna] as km_consegna_noleggio
, [NOL_kmExtra] as km_extra_noleggio
, [NOL_tipoaffidamento] as descr_tipo_affidamento_noleggio
, [NOL_ind_contabilizz] as idx_contabilizzazione_noleggio
, [NOL_giorni_fatt] as giorni_fatturati_noleggio
, [NOL_giorni_fatt_prev] as giorni_fatturati_prev_noleggio
, [NOL_lungo_termine] as is_lungo_termine_noleggio
, [NOL_sede] as sede_noleggio
, [dim_officina] as id_officina
, [NOL_sede_consegna] as sede_consegna_noleggio
, [NOL_pagamento] as pagamento_noleggio
, [NOL_costo_carburante] as costo_carburante_noleggio
, [NOL_costo_veicolo] as costo_veicolo_noleggio
, [NOL_tipo_doc_fatt] as tipo_doc_fatt_noleggio
, [NOL_des_tipo_doc_fatt] as descr_tipo_doc_fatt_noleggio
, [NOL_data_doc_fatt] as dt_doc_noleggio
, [NOL_numero_doc_fatt] as nr_fatt_noleggio
, [NOL_importo_fatt] as imp_fatt_noleggio
, [NOL_TES_DataChiusura] as dt_chiusura_noleggio
, [NOL_TES_NumeroCommessa] as nr_commessa_noleggio
, [NOL_tipo_fatturazione] as tipo_fatturazione_noleggio
, [dim_Commessa] as id_commessa
from FATTI_Officina_Noleggi_INF n;
GO

/****** Object:  View [bi].[OFF_officine]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_officine] as
  select
	o.__ID as id_officina,
	CodiceOfficina codice_officina,
	DescrOfficina descr_officina,
	tro.Descrizione as gruppo_officina
from
	ANAG_Officina_Officina o
left join Tree.Tree_Link_Officina_Officina lo on
	lo.IdDimensione = o.__ID
left join tree.TREE_Officina_Officina tro on
	tro.IDRamo = lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_presenze]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- bi.OFF_presenze source

CREATE view [bi].[OFF_presenze] as

select 
	dim_esecutore as id_esecutore
	, dim_officina as id_officina
	, isnull(dim_tipoesecutore,40) as id_tipo_esecutore --MOD
, HPR_data dt_presenza
, case when HPR_data<GETDATE() then HPR_OreEffettive else null end ore_presenza
, [HPR_OreTeoriche] ore_teoriche_presenza
, [HPR_OreNonProd] ore_nonprod_presenza
, [HPR_OreStraordinario] ore_straordinarie_presenza
, [HPR_OreAssenze] ore_assenza_presenza
, [HPR_OreFerie] ore_ferie_presenza
, [HPR_OrePermessi] ore_permessi_presenza
, [HPR_CostoOrario] imp_costo_orario_presenza
, [HPR_CostoStraordinario] imp_costo_straordinario_presenza
, [HPR_Accettatore] accettatore_presenza

from dbo.FATTI_Officina_Presenze_INF;
GO

/****** Object:  View [bi].[OFF_veicoli]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_veicoli] as 
SELECT
	anag.[__ID] as id_veicolo_officina,
	anag.TES_CodiceAzienda as id_azienda,
	anag.TES_NumeroVeicolo as id_gestionale_veicolo_officina,
	anag.TES_DescrVeicolo as descr_veicolo_officina,
	anag.TES_Targa as targa_veicolo_officina,
	anag.TES_CodiceTipoVeicolo as codice_tipo_veicolo_officina,
	anag.TES_CodiceMarchio codice_marca_veicolo_officina,
	anag.TES_DescrMarchio descr_marca_veicolo_officina,
	tro.descrizione as gruppo_marca_veicolo_officina,
	anag.TES_CodiceModello as codice_modello_veicolo_officina,
	anag.TES_DescrModello as descr_modello_veicolo_officina,
	anag.TES_CodiceVeicolo as codice_veicolo_officina
FROM
	DBA_IntergeaNE.dbo.ANAG_Officina_Veicolo anag
	left outer join ANAG_Officina_Marca o on o.CodiceMarchio = anag.TES_CodiceMarchio 
  	left join Tree.Tree_Link_Officina_GruppoMarca lo on lo.IdDimensione=o.__ID
  	left join tree.TREE_Officina_GruppoMarca tro on tro.IDRamo=lo.IdRamo;
GO

/****** Object:  View [bi].[OFF_voci]    Script Date: 18/04/2025 17:17:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE view [bi].[OFF_voci] as
 select o.__ID as id_voce
 ,OPE_CodiceVoce as codice_voce
 ,OPE_DescrVoce as descr_voce
, tro.Descrizione as gruppo_voce
  from  ANAG_Officina_Voce o
  left join Tree.Tree_Link_Officina_GruppoVoce lo on lo.IdDimensione=o.__ID
  left join tree.TREE_Officina_GruppoVoce tro on tro.IDRamo=lo.IdRamo;
GO

