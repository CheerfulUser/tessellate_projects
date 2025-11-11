from tessellate import Detector
import numpy as np

sector = 39
# for cam in range(1,5):
#     for ccd in range(1,5):
#         print('\n')
#         print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
#         d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')

#         d.collate_filtered_events(save_path=f'/fred/oz335/projects/highlat_transients/events_sig10maxevents5/Sector{sector}',
#                               lower=2,upper=100,min_events=1,max_events=5,psf_like=0.85,lc_sig_max=10,flux_sign=1,
#                               asteroidkiller=True,galactic_latitude=[0,15.],boundarykiller=True,density_score=5)

sectors = np.arange(30,39)

for sector in sectors:
    for cam in range(1,5):
        for ccd in range(1,5):
            print('\n')
            print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
            d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')
            d.collate_filtered_events(save_path=f'/fred/oz335/projects/lowlat_transients/events_sig10/Sector{sector}',
                                    lower=100,upper=2000,min_events=1,max_events=5,psf_like=0.85,lc_sig_max=10,flux_sign=1,
                                    asteroidkiller=True,boundarykiller=True,bkg_level=100,density_score=5,classification='!var')#,galactic_latitude=[0,15.]
            try:
                d.plot_filtered_events(save_path=f'/fred/oz335/projects/lowlat_transients/events_sig10/Sector{sector}',
                                       tess_grid=3,external_phot=False)
            except:
                print('no transients')
