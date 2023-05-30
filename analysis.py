#!/usr/bin/env python3.9
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import os

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# AUTOR: Klára Martinásková

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename : str) -> pd.DataFrame:

    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
                "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
                "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
                "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    #def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    key_list = list(regions.keys())
    val_list = list(regions.values())

    df = pd.DataFrame(columns=headers)
    df["region"] = np.nan

    zip_list = zipfile.ZipFile(filename)

    for year_zip in zip_list.filelist:
        year_folder  = zipfile.ZipFile(zip_list.open(year_zip)) # unzip
        for regionFile in year_folder.filelist:
            with year_folder.open(regionFile.filename) as csv_file: # unzip
                if regionFile.filename[0:2] in val_list: # if region is in regions
                    df_csv = pd.read_csv(csv_file,encoding="cp1250", sep=";", names = headers, low_memory=False) # read csv
                    position = val_list.index(regionFile.filename[0:2])
                    df_csv["region"] = (key_list[position])
                    df = pd.concat([df, df_csv])
    return df

# Ukol 2: zpracovani dat
def parse_data(df : pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    #filename = r"C:\Users\klara\Documents\BTBIO\Inzenyrske_BTBIO\3.semestr\IZV\projekt2\data.zip"
    #df = load_data(filename)

    df_new = df

    df_new["date"] = pd.to_datetime(df["p2a"])

    #dfNew.loc[:, dfNew.columns != "region"| dfNew.columns !=  "p2a" | dfNew.columns != "p1" | dfNew.columns !=  "p2b"| dfNew.columns != "p53" | dfNew.columns != "date"] = dfNew.loc[:, dfNew.columns != "region"| dfNew.columns !=  "p2a" | dfNew.columns != "p1" | dfNew.columns !=  "p2b"| dfNew.columns != "p53" | dfNew.columns != "date"].astype("category")
    df_new = df_new.astype({
            "p5a": "category", "weekday(p2a)" : "category", "p2b" : "category",
            "p6": "category", "p7": "category","p8": "category", 
            "p9": "category", "p10": "category","p11": "category", 
            "p12": "int64", "p13a": "int64", "p13b": "int64", 
            "p13c": "int64","p15": "category","p16": "category", 
            "p37": "category","p17": "category", "p18": "category", 
            "p19": "category", "p20": "category","p21": "category", 
            "p22": "category", "p23": "category", "p24": "category",
            "p27": "category", "p28": "category", "p34":"category", 
            "p36": "category","p39": "category", "p44": "category", 
            "p45a": "category","p48a": "category", "p49": "category", 
            "p50a": "category", "p50b": "category", "p51": "category",
            "p20": "category","p52": "category", "p53": "int64", 
            "p55a": "category","p57": "category", "p58": "category", 
            "h": "category","i": "category", "j": "category", 
            "k": "category","l": "category", "n": "category", 
            "o": "category", "p": "category", "q": "category",
            "r": "category", "s": "category", "t": "category"
        })
    

    df_new["e"] = pd.to_numeric(df_new["e"], errors="coerce")
    df_new["d"] = pd.to_numeric(df_new["d"], errors="coerce")
    df_new["a"] = pd.to_numeric(df_new["a"], errors="coerce")
    df_new["b"] = pd.to_numeric(df_new["b"], errors="coerce")
    df_new["f"] = pd.to_numeric(df_new["f"], errors="coerce")
    df_new["g"] = pd.to_numeric(df_new["g"], errors="coerce")

    df_new = df_new.drop_duplicates(subset=["p1"])

    if(verbose):
        memory_usage_before = df.memory_usage(deep=True, index=True).sum()/1024**2
        memory_usage_after = df_new.memory_usage(deep=True, index=False).sum()/1024**2 
        print("orig_size={} MB".format(memory_usage_before))
        print("new_size={} MB".format(memory_usage_after))
    
    return df_new

# Ukol 3: počty nehod v jednotlivých regionech podle viditelnosti
def plot_visibility(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):

    """
    1 = ve dne + nezhorsena
    2 = ve dne + zhorsena
    3 = ve dne + zhorsena
    4 = v noci + nezhorsena
    5 = v noci + zhorsena
    6 = v noci + nezhorsena
    7 = v noci + zhorsena
    """

    #selected_regions = ["PHA",  "JHM",  "OLK",  "LBK" ]
    df = df[(df.region == "PHA") | (df.region == "JHM") | (df.region == "OLK") | (df.region == "LBK")]

    # replace values
    df["p19"] = df["p19"].replace([1], "ve dne + nezhorsena")
    df["p19"] = df["p19"].replace([3,2], "ve dne + zhorsena") # ve dne + zhorsena
    df["p19"] = df["p19"].replace([4,6], "v noci + nezhorsena") # v noci + nezhorsena
    df["p19"] = df["p19"].replace([5,7], "v noci + zhorsena") # v noci + zhorsena

    df["count"] = 1 # new column count

    df_grouped = df.groupby(["region", "p19"]).agg({"count":"sum"}) # group by region and p19

    df_reset = df_grouped.reset_index() # reset index

    g = sns.catplot( data=df_reset, x="p19", y="count",  hue="region", kind="bar", height=6, aspect=1.9)

    g.set_axis_labels("", "Počet nehod")
    g.set_xticklabels(["Viditelnost: ve dne - nezhoršená", "Viditelnost:  ve dne - zhoršená", "Viditelnost: v noci - nezhoršená", "Viditelnost: v noci - zhoršená"] )
    #g.ax.tick_params(labelsize=10)
    g.fig.suptitle("Počet nehod podle viditelnosti a krajů",
                    fontsize=20, fontdict={"weight": "bold"}) # adding a legend to the top of each bar in the plot based on hue value (region)
    g.fig.subplots_adjust(top=0.9) # adjusting the position of the title
    # show number of accidents in each region
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())), (p.get_x()+0.01, p.get_height() + 700))

    # add horizontal grid lines to the plot
    g.ax.grid(axis="y", linestyle="dashed", alpha=0.5)
    # add legend title
    g.legend.set_title("Kraj")

    if(fig_location):
        if not os.path.exists(fig_location):
            os.makedirs(
                os.path.dirname(fig_location),
                exist_ok=True
            )
        g.savefig(fig_location, dpi=80)
        

    if(show_figure):
        plt.show()

# Ukol4: druh srážky jedoucích vozidel
def plot_direction(df: pd.DataFrame, fig_location: str = None,
                   show_figure: bool = False):

    df = df[(df.region == "PHA") | (df.region == "JHM") | (df.region == "OLK") | (df.region == "LBK")] # select only 4 regions
    """
    1	čelní
    2	boční
    3	z boku
    4	zezadu
    0	nepřichází v úvahu
    """
    
    df = df[df.p7 != 0] # remove rows with p7 = 0 (no car crash)   
    #df["p7"] = df["p7"].replace([0], "nepřichází v úvahu")
    df["p7"] = df["p7"].replace([1], "čelní")
    df["p7"] = df["p7"].replace([2], "boční")
    df["p7"] = df["p7"].replace([3], "z boku")
    df["p7"] = df["p7"].replace([4], "zezadu")

    df["p7"] = df["p7"].astype("string") # convert to string

    df["count"] = 1 # for counting accidents

    df_grouped = df.groupby(["region","p7", df.date.dt.month]).agg({"count":"sum"}) # group by region, p7 and month
  
    df_reset = df_grouped.reset_index() 
    
    ### plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    fig.suptitle("Počet nehod podle kraje a druhu srážky", fontsize=16, fontweight="bold")
    months = ["leden", "únor", "březen", "duben", "květen", "červen", "červenec", "srpen", "září", "říjen", "listopad", "prosinec"]

    # JMK
    sns.barplot(ax=axes[0,0], data=df_reset.loc[df_reset["region"] == "JHM"], x="date", y="count", hue="p7")
    axes[0,0].set_title("Jihomoravský kraj")
    axes[0,0].set_xticklabels(months, rotation=45) # set labels at x axis as months
    axes[0,0].set_xlabel("Měsíc")
    axes[0,0].set_ylabel("Počet nehod")
    axes[0,0].legend().set_visible(False) # legend not visible in axes

    # LBK
    sns.barplot(ax=axes[0,1], data=df_reset.loc[df_reset["region"] == "LBK"], x="date", y="count", hue="p7")
    axes[0,1].set_title("Liberecký kraj")
    axes[0,1].set_xticklabels(months, rotation=45)
    axes[0,1].set_xlabel("Měsíc")
    axes[0,1].set_ylabel("Počet nehod")
    axes[0,1].legend().set_visible(False)

    # OLK
    sns.barplot(ax=axes[1,0], data=df_reset.loc[df_reset["region"] == "OLK"], x="date", y="count", hue="p7")
    axes[1,0].set_title("Olomoucký kraj")
    axes[1,0].set_xticklabels(months, rotation=45)
    axes[1,0].set_xlabel("Měsíc")
    axes[1,0].set_ylabel("Počet nehod")
    axes[1,0].legend().set_visible(False)

    # PHA
    sns.barplot(ax=axes[1,1], data=df_reset.loc[df_reset["region"] == "PHA"], x="date", y="count", hue="p7")
    axes[1,1].set_title("Praha")
    axes[1,1].set_xticklabels(months, rotation=45)
    axes[1,1].set_xlabel("Měsíc")
    axes[1,1].set_ylabel("Počet nehod")
    axes[1,1].legend().set_visible(False)


    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5), title="Druh srážky")
    fig.tight_layout() # not to cover title with xticks


    if(fig_location):
        if not os.path.exists(fig_location):
            os.makedirs(
                os.path.dirname(fig_location),
                exist_ok=True
            )
        fig.savefig(fig_location, dpi=100, bbox_inches="tight")
        

    if(show_figure):
        fig.show()

# Ukol 5: Následky v čase
def plot_consequences(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
        
    """
    p13a	usmrceno osob
    p13b	těžce zraněno osob
    p13c	lehce zraněno osob

    """

    regions = ["PHA",  "JHM",  "OLK",  "LBK" ] # extract rows with specified regions
    df2 = df[(df.region == "PHA") | (df.region == "JHM") |
                (df.region == "OLK") | (df.region == "LBK")]

    # extract rows with specified columns
    df2 =  df2.loc[((df2["p13a"]!=0) | (df2["p13b"]!=0) | (df2["p13c"]!=0))]
    conditions = [df2["p13a"] >= 1, df2["p13b"] >= 1, df2["p13c"] >= 1]

    values = ["Usmrcení","Těžké zranění","Lehké zranění"] # values for new column
    df2["následky"] = np.select(conditions, values) # create new column with values
    df2["následky"] = df2["následky"].astype("string")
    cross_table_df = pd.crosstab(
        index=[df2["region"], df2["date"]],
        columns=df2["následky"])
    
    data = cross_table_df.unstack(level=0).resample("M").sum().stack(level=1).swaplevel(1, 0) # unstacking and resampling

    data_grouped = data.groupby(["region", "date"]).sum().reset_index() # group by region and date

    # melting data
    data_melted = pd.melt(data_grouped, id_vars=["region", "date"]).rename(
        columns={"následky": "Následky nehody"})

    # plots
    plot = sns.FacetGrid(data_melted, col="region",
                            hue="Následky nehody", height=4,
                            sharex=False, sharey=False, col_wrap=2)
    plot = plot.map(sns.lineplot, "date", "value", ci=None).add_legend()
    plot.set_axis_labels("Datum vzniku nehody", "Počet nehod")

    j = 0

    for ax in plot.fig.get_axes(): # setting titles
        ax.set_title("Kraj: " + regions[j])
        j += 1

    if(show_figure):
        plt.show()
    if (fig_location):
        if not os.path.exists(fig_location):
            os.makedirs(
                os.path.dirname(fig_location),
                exist_ok=True
            )
        plot.savefig(fig_location, dpi=80)

if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni 
    # funkce.
    df = load_data(r"C:\Users\klara\Documents\BTBIO\Inzenyrske_BTBIO\3.semestr\IZV\projekt2\data.zip")
    df2 = parse_data(df, True)

    # save df2 to pickle
    df2.to_pickle(r"C:\Users\klara\Documents\BTBIO\Inzenyrske_BTBIO\3.semestr\IZV\projekt_final\df2.pkl")
   


    
    plot_visibility(df2, r"C:\Users\klara\Documents\BTBIO\Inzenyrske_BTBIO\3.semestr\IZV\projekt2\01_visibility.png", True)
    plot_direction(df2, r"C:\Users\klara\Documents\BTBIO\Inzenyrske_BTBIO\3.semestr\IZV\projekt2\02_direction.png", True)
    plot_consequences(df2, r"C:\Users\klara\Documents\BTBIO\Inzenyrske_BTBIO\3.semestr\IZV\projekt2\03_consequences.png", True)


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
