from data.market_data import get_expiries, get_spot_price
from surface.vol_surface import build_surface, plot_vol_heatmap, interplote_surface, plot_vol_surface
import pandas as pd


S = get_spot_price("SPY")
expiries = get_expiries("SPY")
surface_df = build_surface("SPY", expiries[:10], S)
print(surface_df.head())
print(surface_df.shape)
heatmap_data =interplote_surface(surface_df)
#print(heatmap_data.isna().sum().sum())
plot_vol_heatmap(heatmap_data)
plot_vol_surface(heatmap_data)
