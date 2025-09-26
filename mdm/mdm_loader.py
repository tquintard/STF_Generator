from modules.loader import load_excel
from config.settings import MDM_FILE

def load_mda():
    mda_df = load_excel(MDM_FILE, sheet_name="Attribute")
    facet_labels_df = load_excel(MDM_FILE, sheet_name="Facet")

    if not mda_df.empty and {"MUDU", "Facet"}.issubset(mda_df.columns):
        # On garde uniquement les attributs présents dans la DF sélectionnée
        #mda_filtered = mda_df[mda_df["MUDU"].isin(selected.columns)]

        # Liste unique des facettes dans l'ordre du fichier Excel
        #facets = mda_df["Facet"].dropna().unique().tolist()

        # Mapping Facet -> Label si la feuille "Facet" existe
        facet_labels = dict(zip(facet_labels_df["Name"], facet_labels_df["Label"]))

    return mda_df, facet_labels