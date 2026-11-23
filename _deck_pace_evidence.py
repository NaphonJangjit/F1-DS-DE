"""Reproduce the holdout baselines and race-distance pace chart for the deck."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import _fforest as ff

ROOT = Path(__file__).resolve().parent
features = ["Prev_Duration", "Prev_Sector_Duration", "Rep_Median3"]
anchor = "Rep_Prev_Duration"

def inputs(frame):
    return frame[features].div(frame[anchor], axis=0).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)

def main():
    frames = {year: ff._load_year(year, ["Duration"]) for year in (2023, 2024, 2025)}
    train = pd.concat([frames[2023], frames[2024]])
    test = frames[2025]
    for frame in (train, test):
        assert frame[anchor].gt(0).all()
    model = LinearRegression().fit(inputs(train), train.Duration / train[anchor] - 1)
    estimates = {"Repeat clean sector": test[anchor], "Linear pace model": test[anchor] * (1 + model.predict(inputs(test)))}
    errors = {k: float((v - test.Duration).abs().mean()) for k, v in estimates.items()}
    saved = json.loads((ROOT / "model/v3/fforest_results.json").read_text())
    def find_breakdown(obj):
        if isinstance(obj, dict):
            if obj.get("anchor") == anchor and "breakdown" in obj:
                return obj["breakdown"]
            for v in obj.values():
                found = find_breakdown(v)
                if found: return found
    score = find_breakdown(saved)["overall"]
    assert len(test) == score["n"]
    errors["Two-stage forest"] = score["mae_s"]
    raw = ff.load_raw_year(2025)
    keys = ["Session_Key","Driver","Lap"]
    raw["green_dry"] = raw.Track_Status.eq("GREEN") & raw.Rainfall.eq(0) & raw.Tyre.isin(["HARD","SOFT","MEDIUM"])
    laps = raw.groupby(keys).agg(time=("Duration","sum"), count=("Duration","count"), sectors=("Sector","nunique"), rows=("Sector","size"), clean=("green_dry","all"), pit=("Will_Box","max"), outlap=("Prev_Lap_Pitted","max")).reset_index()
    laps["finish"] = laps.groupby("Session_Key").Lap.transform("max")
    laps = laps.loc[(laps["count"]==3) & (laps.sectors==3) & (laps.rows==3) & laps.clean & laps.pit.eq(0) & laps.outlap.eq(0) & laps.time.gt(0)].copy()
    laps["bin"] = np.minimum(4, np.ceil(5*laps.Lap/laps.finish).astype(int)-1)
    laps["delta"] = 100*(laps.time / laps.groupby(["Session_Key","Driver"]).time.transform("median")-1)
    race = laps.groupby(["Session_Key","bin"]).delta.median().unstack().dropna()
    med = race.median()
    quant = race.quantile([.25,.75])
    n = int(laps.Session_Key.isin(race.index).sum())
    fig, ax = plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    x=np.arange(5)
    ax[0].fill_between(x,quant.loc[.25],quant.loc[.75],color="#abcde3",label="Middle 50% of races")
    ax[0].plot(x,med,"o-",color="#13597a",label="Median race")
    ax[0].axhline(0,color="gray",lw=.8)
    ax[0].set(xticks=x,xticklabels=["0–20","20–40","40–60","60–80","80–100"],xlabel="Race distance (%)",ylabel="Difference from driver’s race median (%)",title="Dry green lap time across the race")
    ax[0].legend()
    b=ax[1].barh(list(errors),list(errors.values()),color=["#74838c","#37839d","#bf332e"])
    ax[1].bar_label(b,fmt="%.3f",padding=3)
    ax[1].set(xlim=(0,max(errors.values())*1.2), xlabel="MAE (seconds per sector)", title="2025 sector prediction error")
    fig.savefig(ROOT/"figures/pace_holdout_2025.png",dpi=160)
    print(json.dumps(dict(train=len(train),test=len(test),errors=errors,laps=n,races=len(race),median=med.tolist(),q25=quant.loc[.25].tolist(),q75=quant.loc[.75].tolist(),coef=model.coef_.tolist(),intercept=model.intercept_)))
if __name__ == "__main__":
    main()

