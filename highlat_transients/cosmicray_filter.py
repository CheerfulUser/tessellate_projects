import pandas as pd
import numpy as np
import os
from tessellate import Detector
from tessellate.tools import _Save_space

path = f'/fred/oz335/projects/highlat_transients/events_sig10maxevents5'
event_frame_buffer = 20
calc_window = 60

def iterative_clip(flux, ind, tol=5e-2,max_recursion=2):
    mask = ind.copy()
    current_std = np.std(flux[mask])
    current_med = np.median(flux[mask])

    done = False
    count = 0
    while (done == False) & (count<=max_recursion):
        masked_flux = flux[mask]
        brightest_idx = np.argmax(masked_flux)
        mask_indices = np.where(mask)[0]
        new_mask = mask.copy()
        new_mask[mask_indices[brightest_idx]] = False

        new_std = np.std(flux[new_mask])
        frac_change = (current_std - new_std) / current_std

        if frac_change < tol:
            done=True
        else:
            mask = new_mask
            current_std = new_std
            current_med = np.median(flux[mask])
        count += 1

    return mask, current_med, current_std

for s in range(39,40):
    print('\n')
    print('\n')
    print(f'Sector{s}')

    events = pd.read_csv(f'{path}/Sector{s}/events.csv')
    _Save_space(f'{path}/Sector{s}_CosmicRays')

    # -- Adds in a variable to see how individual frames each event was actually detected in (not through light curve significance) -- #
    print('     getting n_triggers')
    n_triggers = []
    for cam in range(1,5):
        for ccd in range(1,5):

            print(cam,ccd)
            ccd_events = events[(events.camera==cam)&(events.ccd==ccd)]
            
            if len(ccd_events)>0:
                d = Detector(sector=s,cam=cam,ccd=ccd)
                for i,ev in ccd_events.iterrows():
                    if ev.cut != d.cut:
                        d.cut = ev.cut
                        d._gather_results(cut=ev.cut,sources=True,events=False,objects=False)
                        d.cut = ev.cut
                    objsources = d.sources[d.sources.objid==ev.objid]
                    objsources = objsources[(objsources['frame']>=ev.frame_start)&(objsources['frame']<=ev.frame_end)]
                    n_triggers.append(len(objsources))

    events.insert(len(events.columns)-2,'n_triggers',n_triggers)
    events.to_csv(f'{path}/Sector{s}/events.csv',index=False)

    # -- In this case, all events which only have 2 points that are "significant" but are not consecutive are not real -- #
    print('     moving case')

    cosmicrays = events[(events.n_detections==2)&(events.frame_duration>2)]

    # -- Move all cosmicrays found through this method -- #
    for i,ev in cosmicrays.iterrows():
        cam = ev['camera']
        ccd = ev['ccd']
        cut = ev['cut']
        objid = ev['objid']
        eventid = ev['eventid']

        os.system(f'mv {path}/Sector{s}/S{s}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png {path}/Sector{s}_CosmicRays')

    events = pd.concat([events, cosmicrays]).drop_duplicates(keep=False)

    # -- This now looks at all remaining events with just one actual trigger and only two points supposedly significant.
    #    This corrects for psf centroid affecting true LC, by only keeping events that still have two significant points consecutively -- #
    print('     moving case')
    potential = events[(events.n_triggers==1)&(events.n_detections==2)]
    real = pd.DataFrame()
    d = Detector(sector=s,cam=1,ccd=1)
    d.cut = 100
    for i, ev in potential.iterrows():
        s = ev.sector
        cam = ev.camera
        ccd = ev.ccd
        cut = ev.cut
        objid = ev.objid
        eventid=ev.eventid

        if (d.cam != cam) |(d.ccd != ccd) | (d.cut != cut) :
            d = Detector(sector=s,cam=cam,ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')
            d._gather_data(cut=cut,flux=True,bkg=False,wcs=False,mask=False,ref=False) 
            d._gather_results(cut=cut,sources=False,objects=False)

        t,f = d.event_lc(cut=cut,objid=objid,eventid=eventid,frame_buffer=10000)[0]

        start = ev['frame_start']
        end = ev['frame_end']

        fs = start - event_frame_buffer
        fe = end + event_frame_buffer
        if fs < 0:
            fs = 0
        if fe > len(t):
            fe = len(t) - 1 
        bs = fs - calc_window
        be = fe + calc_window
        if bs < 0:
            bs = 0
        if be > len(t):
            be = len(t) - 1 

        frames = np.arange(0,len(t))

        ind = ((frames > bs) & (frames < fs)) | ((frames < be) & (frames > fe))

        mask,med,std = iterative_clip(f,ind)

        lc_sig = (f - med) / std

        sig_event = lc_sig[start:end+1]

        if len(sig_event[sig_event>3]) == 1:
            real = pd.concat([real,potential.loc[[i]]],ignore_index=True)

    # -- Move all cosmicrays found through this method -- #
    for i,ev in real.iterrows():
        cam = ev['camera']
        ccd = ev['ccd']
        cut = ev['cut']
        objid = ev['objid']
        eventid = ev['eventid']

        os.system(f'mv {path}/Sector{s}/S{s}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png {path}/Sector{s}_CosmicRays')

    # -- Restrict events csv accordingly -- #
    events = pd.concat([events, real]).drop_duplicates(keep=False)
    events.to_csv(f'{path}/Sector{s}/events.csv',index=False)

    # -- Build cosmicrays dataframe and save out -- #
    cosmicrays = pd.concat([cosmicrays,real],ignore_index=True)

    if os.path.exists(f'{path}/Sector{s}_CosmicRays/events.csv'):
        pre = pd.read_csv(f'{path}/Sector{s}_CosmicRays/events.csv')
        cosmicrays = pd.concat([cosmicrays,pre],ignore_index=True).drop_duplicates(keep='first')

    cosmicrays = cosmicrays.sort_values(by=["camera", "ccd", "cut", "objid",'eventid'])
    cosmicrays.to_csv(f'{path}/Sector{s}_CosmicRays/events.csv',index=False)



