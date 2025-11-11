import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tessellate import Detector
from tessellate.tools import _Save_space
from tessellate.external_photometry import _delve_objects, get_gaia, check_gaia
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

for s in range(27,40):
    print(s)
    path = f'/fred/oz335/projects/highlat_transients/events_sig10maxevents5'
    events = pd.read_csv(f'{path}/Sector{s}/events.csv')
    _Save_space(f'{path}/Sector{s}_Stars')

    stars = pd.DataFrame()
    for i, ev in events.iterrows():
        sources = _delve_objects(ev['ra'],ev['dec'],size=10/60**2)
        gaia = get_gaia(ev['ra'],ev['dec'],10/60**2)
        if (gaia is not None) & (sources is not None):
            sources = check_gaia(sources,gaia)

        
        if len(sources) > 0:
            coords = SkyCoord(sources["ra"].values * u.deg, sources["dec"].values * u.deg, frame="icrs")
            ref = SkyCoord(ev['ra'] * u.deg, ev['dec'] * u.deg, frame="icrs")
            sources["dist_arcsec"] = coords.separation(ref).deg*3600

            if (sources[sources['dist_arcsec']<6]['star'].min() == 1) & (sources[sources['dist_arcsec']<6]['star'].max() == 1):
                os.system(f"mv {path}/Sector{s}/S{s}C{ev['camera']}C{ev['ccd']}C{ev['cut']}O{ev['objid']}E{ev['eventid']}.png {path}/Sector{s}_Stars")
                stars = pd.concat([stars,events.loc[i]],ignore_index=True)
    
    events = pd.concat([events,stars],ignore_index=True).drop_duplicates(keep=False)
    events.to_csv(f'{path}/Sector{s}/events.csv',index=False)

    stars.to_csv(f'{path}/Sector{s}_Stars/events.csv',index=False)

    

