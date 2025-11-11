from tessellate import Detector
import numpy as np

sectors = np.arange(28,39,dtype=int)

for sector in sectors:
    # for cam in range(1,5):
    #     for ccd in range(1,5):
    #         print('\n')
    #         print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
    #         d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')

    #         d.collate_filtered_events(save_path=f'/fred/oz335/projects/exoplanets/sig5/Sector{sector}',
    #                             lower=10,max_events=20,lc_sig_max=5,flux_sign=-1,bkg_std=50,
    #                             boundarykiller=True,classification='!var')

    for cam in range(1,5):
        for ccd in range(1,5):
            print('\n')
            print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
            d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')
            d.collate_filtered_events(save_path=f'/fred/oz335/projects/exoplanets/sig5/Sector{sector}',
                                      lower=1,max_events=20,lc_sig_max=5,flux_sign=-1,bkg_level=100,
                                      boundarykiller=True,classification='!var')
            d.plot_filtered_events(save_path=f'/fred/oz335/projects/exoplanets/sig5/Sector{sector}',
                                   tess_grid=3,external_phot=False,ref=True,latex=False)     
            d.lc_filtered_events(save_path=f'/fred/oz335/projects/exoplanets/sig5/Sector{sector}',ref=True)
            
               
