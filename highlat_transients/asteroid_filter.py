import pandas as pd
import os
from tessellate.tools import _Save_space

path = f'/fred/oz335/projects/highlat_transients/events_sig10maxevents5'

for s in range(39,40):
    print('\n')
    print('\n')
    print(f'Sector{s}')
    events = pd.read_csv(f'{path}/Sector{s}/events.csv')

    asteroids = events[(events['com_motion']>0.4)&(events['gaussian_score']>0.7)]
        
    _Save_space(f'{path}/Sector{s}_Asteroids')

    if len(asteroids)>0:
        for i, ev in asteroids.iterrows():
            cam = ev['camera']
            ccd = ev['ccd']
            cut = ev['cut']
            objid = ev['objid']
            eventid = ev['eventid']

            os.system(f'mv {path}/Sector{s}/S{s}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png {path}/Sector{s}_Asteroids')
        
        events = pd.concat([events, asteroids]).drop_duplicates(keep=False)
        events.to_csv(f'{path}/Sector{s}/events.csv',index=False)

        if os.path.exists(f'{path}/Sector{s}_Asteroids/events.csv'):
            pre = pd.read_csv(f'{path}/Sector{s}_Asteroids/events.csv')
            asteroids = pd.concat([asteroids,pre],ignore_index=True).drop_duplicates(keep='first')

        asteroids = asteroids.sort_values(by=["camera", "ccd", "cut", "objid",'eventid'])
        asteroids.to_csv(f'{path}/Sector{s}_Asteroids/events.csv',index=False)

