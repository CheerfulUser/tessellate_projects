from tessellate import Detector

sector = 39
# for cam in range(1,5):
#     for ccd in range(1,5):
#         print('\n')
#         print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
#         d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')

#         d.collate_filtered_events(save_path=f'/fred/oz335/projects/highlat_transients/events_sig10maxevents5/Sector{sector}',
#                               lower=2,upper=50,min_events=2,max_events=5,psf_like=0.85,lc_sig_max=10,flux_sign=1,
#                               asteroidkiller=True,galactic_latitude=15.,boundarykiller=True,density_score=5)

for cam in range(1,5):
    for ccd in range(1,5):
        print('\n')
        print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
        d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')

        d.plot_filtered_events(save_path=f'/fred/oz335/projects/highlat_transients/events_sig10maxevents5/Sector{sector}',tess_grid=3)        
