import numpy as np
import pandas as pd
import os
from glob import glob
from copy import deepcopy

from tessellate import Detector
import tessreduce as tr
from tessreduce import * 

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from astropy import coordinates
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord

from astroquery.vizier import Vizier
from astroquery.exoplanet_orbit_database import ExoplanetOrbitDatabase



class ExoHunter:

    def __init__(self,sector,data_path,save_path,build_path=True,run=True):
        self.sector = sector
        self.build_path = build_path
        self.data_path = data_path
        self.save_path = save_path
        if run:
            self.main()


    def build_directory(self):
        path = f'{self.save_path}/Sector{self.sector}/'
        os.makedirs(path,exist_ok=True)
        for ext in ['lc','figs','norm_lcs','norm_figs','potential_exo']:
            os.makedirs(path + ext,exist_ok=True)
        for i in ['candidates', 'known_exos', 'possibles']:
            os.makedirs(path + ext + '/' + i + '/lcs',exist_ok=True)
            os.makedirs(path + ext + '/' + i + '/figs',exist_ok=True)


    def panoptes_test(self,sector=None,cams=[1,2,3,4],ccds=[1,2,3,4]):
        # sectors = [29]
        # cams = [1,2,3,4]
        # ccds = [1,2,3,4]
        if type(sector) != type(None):
            self.sector = sector
        cuts = np.arange(1,16)+ 1 

        good_tags = []
        good_figs = []
        good = None
        for cam in cams:
            for ccd in ccds:
                for cut in cuts:
                    try:
                        path = f'/fred/oz335/TESSdata/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of16/'
                        figs_full = glob(path+'figs/**/*.png', recursive=True)
                        figs = np.array([f.split('/')[-1].split('.png')[0] for f in figs_full])
                        figs_full = np.array(figs_full)
                        data = pd.read_csv(path + 'detected_events.csv')
                        sig_ind = (abs(data['lc_sig'].values) >= 2.5) & (data['max_sig'].values >= 3) #& (data['lc_sig_med'].values >= 2) #& (data['sig_av'].values >= 2.5)
                        type_ind = (data['Type'].values =='0') #& (data['Type'].values != 'Asteroid')
                        num_ind = data['total_events'].values < 4
                        star_ind = data['GaiaID'].values == 0
                        data.loc[np.isnan(data['peak_power'].values),'peak_power'] = 0
                        peak_ind = data['peak_power'].values == 0
                        freq_ind = 1/data['peak_freq'].values
                        freq_ind = (freq_ind <= 14) | (freq_ind >= 17)
                        sign_ind = data['flux_sign'].values < 0
                        dur_ind = (data.frame_end.values - data.frame_start.values) > 2
                        bkg_ind = data.bkg_level.values < 100
                        ind = sig_ind & sign_ind  #& dur_ind & type_ind  & bkg_ind & peak_ind  & num_ind & peak_ind & star_ind & sign_ind

                        sample = data.iloc[ind & dur_ind]
                        gind = []
                        for i in range(len(sample)):
                            good_tag = f'Sec{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_object{sample.objid.values[i]}_event{sample.eventID.values[i]}of{sample.total_events.values[i]}'
                            good_tags += [good_tag]
                            fig_name = figs_full[good_tag == figs]
                            if len(fig_name) > 0:
                                gind += [i]
                                good_figs += [fig_name]
                        gind = np.array(gind)
                        if type(good) == type(None):
                            good = sample.iloc[gind]
                        else:
                            good = pd.concat([good,sample.iloc[gind]])
                    except:
                        pass

        good_figs = np.array(good_figs)

        ff = []
        for file in good_figs:
            p, f = file[0].split('figs')
            f = f.split('/')[-1]
            ff += [f]
        good['fig_names'] = ff


        # ind = (good['flux_sign'].values < 0) & (abs(good['lc_sig_med'].values) >= 2.5)  & (abs(good.bkg_level.values) < 100) & (good['Type'].values =='0')  & (good['peak_power'].values == 0) & (good['total_events'].values < 30) & ((good.frame_end.values - good.frame_start.values) >= 3) #& (~good['variable'].values) # #
        ind = (good['flux_sign'].values < 0) & (abs(good['lc_sig_med'].values) >= 2.5)  & (abs(good.bkg_level.values) < 100) & (good['Type'].values =='0') & (good['total_events'].values < 30) & ((good.frame_end.values - good.frame_start.values) >= 3) #& (~good['variable'].values) # #
        sub = good.iloc[ind]
        #unique,indo = np.unique(sub['objid'].values,return_index=True)
        #sub = sub.iloc[indo]
        figs = good_figs[ind]#[indo]
        figs = figs.flatten()

        path = f'{self.data_path}/Sector{self.sector}/'
        paths = ['figs/','lc/']

        # for ext in paths:
        #    print(path + ext)
        #    call = f'mkdir {path + ext}'
        #    os.system(call)

        for fig in figs:#[ind]:
            call = f'cp {fig} {path + paths[0]}'
            os.system(call)
        
        for file in figs:#[ind]:
            p, f = file.split('figs')
            f = f.split('/')[-1].split('.png')[0] + '.csv'
            #    call = f'cp {p+ 'lcs/'+f} {path + paths[1]}'
            call = f'cp {p}lcs/{f} {path + paths[1]}'
            os.system(call)

        name = 'planets'
        sub.to_csv(f'{path+name}.csv',index=False)


    def get_ref_counts(self):
        data_path = '/fred/oz335/TESSdata'
        # sector = 30
        # camera = 3
        # ccd = 4
        n = 4

        exos = pd.read_csv(f'{self.save_path}/Sector{self.sector}/planets.csv')

        sec = exos['sector'].values
        cam = exos['camera'].values
        ccd = exos['ccd'].values
        cut = exos['cut'].values
        x = (exos['x_source'].values + 0.5).astype(int)
        y = (exos['y_source'].values + 0.5).astype(int)

        flux = []
        for i in range(len(sec)):
            d = Detector(sector=sec[i],cam=cam[i],ccd=ccd[i],n=n,data_path=data_path)
            name = f'{d.path}/Cut{cut[i]}of{d.n**2}/sector{d.sector}_cam{d.cam}_ccd{d.ccd}_cut{cut[i]}_of{d.n**2}_Ref.npy'
            ref = np.load(name)
            flux += [np.nansum(ref[y[i]-1:y[i]+2,x[i]-1:x[i]+2])]

        exos['ref_counts'] = flux
        exos.to_csv(f'{self.save_path}/Sector{self.sector}/planet_ref.csv')
    

    def classify_transit_type(self):
        def consecutive_points(data, stepsize=3):
            return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


        path = f'{self.data_path}/Sector{self.sector}/'
        save_path = f'{self.save_path}/Sector{self.sector}/'
        files = glob(save_path+'norm_lcs/*.csv')


        def parabola(x, a, b, c):
            p = a*(x - b)*(x - c)
            return p


        def vet_transit(lc):
            try:
                lc = np.array([lc['mjd'].values,lc['counts'].values])
            except:
                pass
            m, m, lcstd = sigma_clipped_stats(lc[1])
            lc_sig = (lc[1]-1)/lcstd

            siglim = -5
            sig_ind = np.where(lc_sig <= siglim)[0]
            segments = consecutive_points(sig_ind)
            duration = []
            depth = []
            i = 0
            for s in segments:
                if len(s) > 2:
                    try:
                        y = lc[1,s[0]-4:s[-1]+5] - 1
                        x = np.arange(s[0]-4,s[-1]+5)
                        bds = ((0,s[0]-10,s[0]-10),(10,s[-1]+11,s[-1]+11))
                        fit_params, pcov = curve_fit(parabola, x, y, bounds=bds)
                        fit = parabola(x,*fit_params)
                        ss_res = np.sum((y - fit) ** 2)
                        # total sum of squares
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        # r-squared
                        r2 = 1 - (ss_res / ss_tot)
                        #print(np.round(r2,2))
                        if r2 >= 0.5:
                            duration += [abs(lc[0,int(fit_params[2]+0.5)]-lc[0,int(fit_params[1]+0.5)])]
                            l = lc[1,s]
                            #plt.figure()
                            #plt.plot(x,y)
                            #plt.plot(x,parabola(x,*fit_params))
                            #print(s)
                            depth += [-np.min(parabola(x,*fit_params))]
                    except:
                        pass
                i += 1
            duration = np.array(duration)
            depth = np.array(depth)
            matched_depth = (abs((depth[:,np.newaxis] - depth[np.newaxis,:])) < 5*lcstd)
            matched_duration = (abs((duration[:,np.newaxis] - duration[np.newaxis,:])) < 5/24)

            if len(depth) > 1:
                std = np.std(depth)
                if (~matched_depth).any():
                    if (matched_duration).all():
                        likely = 'EB'
                    else:
                        likely = 'MP'
                else:
                    likely = 'MT'
            elif len(depth) == 0:
                likely = 'No'

            else:
                likely = 'ST'

            return likely

        exo = pd.read_csv(path+'events.csv')
        exo['exo_class'] = 'No'
        for file in files:
            lc = pd.read_csv(file)
            likely = vet_transit(lc)
            _,cam,ccd,cut = file.split('/')[-1].split('C')
            cut,obj = cut.split('O')
            obj = obj.split('_')[0]
            ind = (exo['camera'].values==int(cam)) & (exo['ccd'].values==int(ccd)) & (exo['cut'].values==int(cut)) & (exo['objid'].values==int(obj))
            
            exo.loc[ind,'exo_class'] = likely

        exo.to_csv(save_path+'events_likely.csv',index=False)

        # plt.figure()
        # plt.hist(likely)
        # plt.savefig(f'{self.save_path}/Sector{self.sector}/likely_hist',dpi=750)


    def flatten_lcs(self):
        path = f'{self.data_path}/Sector{self.sector}/'
        save_path = f'{self.save_path}/Sector{self.sector}/'
        files = glob(path+'lcs/*.csv')
        lc = pd.read_csv(files[0])
        lc = np.array([lc['mjd'].values,lc['counts'].values])
        exos = pd.read_csv(path+'events.csv')


        def auto_tail(lc,mask,err = None):
            if err is not None:
                higherr = sigma_clip(err,sigma=2).mask
            else:
                higherr = False
            masks = Identify_masks(mask*1)
            med = np.nanmedian(lc[1][~mask & ~higherr])
            std = np.nanstd(lc[1][~mask & ~higherr])

            if lc.shape[1] > 4000:
                tail_length = 50
                start_length = 10

            else:
                tail_length = 5
                start_length = 1

            for i in range(len(masks)):
                m = np.argmax(lc[1]*masks[i])
                sig = (lc[1][m] - med) / std
                median = np.nanmedian(sig[sig>0])
                if median > 50:
                    sig = sig / 100
                    #sig[(sig < 1) & (sig > 0)] = 1
                if sig > 20:
                    sig = 20
                if sig < 0:
                    sig = 0
                masks[i][int(m-sig*start_length-10):int(m+tail_length*sig+10)] = 1
                masks[i] = masks[i] > 0
            summed = np.nansum(masks*1,axis=0)
            mask = summed > 0 
            return ~mask

        def detrend_stellar_var(lc,err=None,Mask=None,variable=False,sig = None, 
                                sig_up = 100, sig_low = 2, tail_length='auto',normalise=True):
            """
            Removes all long term stellar variability, while preserving flares. Input a light curve 
            with shape (2,n) and it should work!

            Parameters
            ----------
            lc : array_like, optional
                lightcurve with the shape of (2,n), where the first index is time and the second is 
                flux. The default is None.
            err : array_like, optional
                Flux error to be used in weighting of events of size (n,). The default is None.
            Mask : array_like, optional
                1d one dimensional mask of the lightcurve to not be included in the detrending. The default is None.
            variable : bool, optional
                Determine whether the object is variable. The default is False.
            sig : float, optional
                Significance of the event before it gets excluded. The default is None.
            sig_up : Float, optional
                Upper sigma clip value . The default is 5.
            sig_low : Float, optional
                Lower sigma clip value. The default is 10.
            tail_length : str OR int, optional
                Option for setting the buffer zone of points after the peak. If it is 'auto' it 
                will be determined through functions, but if its an int then it will take the given 
                value as the buffer tail length for fine tuning. The default is ''.

            Raises
            ------
            ValueError
                "tail_length must be either 'auto' or an integer".

            Returns
            -------
            detrend : array_like
                Lightcurve with the stellar trends subtracted.

            """

            # Make a smoothing value with a significant portion of the total 
            nonan = np.isfinite(lc[1])
            lc = lc[:,nonan]

            if (err is not None):
                err = err[nonan]

            trends = np.zeros(lc.shape[1])
            break_inds = Multiple_day_breaks(lc)
            #lc[Mask] = np.nan

            if variable:
                size = int(lc.shape[1] * 0.1)
                if size % 2 == 0: size += 1

                finite = np.isfinite(lc[1])
                smooth = savgol_filter(lc[1,finite],size,1)		
                # interpolate the smoothed data over the missing time values
                f1 = interp1d(lc[0,finite], smooth, kind='linear',fill_value='extrapolate')
                smooth = f1(lc[0])
                #mask = sig_err(lc[1]-smooth,err,sig=sig)
                mask = sigma_clip(lc[1]-smooth,sigma=sig,sigma_upper=sig_up,
                                    sigma_lower=sig_low,masked=True).mask
            else:
                mask = sig_err(lc[1],err,sig=sig)

            ind = np.where(mask)[0]
            masked = lc.copy()
            # Mask out all peaks, with a lead in of 5 frames and tail of 100 to account for decay
            # todo: use findpeaks to get height estimates and change the buffers accordingly
            if type(tail_length) == str:
                if tail_length == 'auto':

                    m = auto_tail(lc,mask,err)
                    masked[:,~m] = np.nan


                else:
                    if lc.shape[1] > 4000:
                        tail_length = 100
                        start_length = 1
                    else:
                        tail_length = 10
                    for i in ind:
                        masked[:,i-5:i+tail_length] = np.nan
            else:
                tail_length = int(tail_length)
                if type(tail_length) != int:
                    raise ValueError("tail_length must be either 'auto' or an integer")
                for i in ind:
                    masked[:,i-5:i+tail_length] = np.nan


            ## Hack solution doesnt need to worry about interpolation. Assumes that stellar variability 
            ## is largely continuous over the missing data regions.
            #f1 = interp1d(lc[0,finite], lc[1,finite], kind='linear',fill_value='extrapolate')
            #interp = f1(lc[0,:])

            # Smooth the remaining data, assuming its effectively a continuous data set (no gaps)
            size = int(lc.shape[1] * 0.005)
            if size % 2 == 0: 
                size += 1
            for i in range(len(break_inds)-1):
                section = lc[:,break_inds[i]:break_inds[i+1]]
                finite = np.isfinite(masked[1,break_inds[i]:break_inds[i+1]])
                smooth = savgol_filter(section[1,finite],size,1)

                # interpolate the smoothed data over the missing time values
                f1 = interp1d(section[0,finite], smooth, kind='linear',fill_value='extrapolate')
                trends[break_inds[i]:break_inds[i+1]] = f1(section[0])
            # We now have a trend that should remove stellar variability, excluding flares.
            detrend = deepcopy(lc)
            if normalise:
                detrend[1,:] = lc[1,:] / trends
            else:
                detrend[1,:] = lc[1,:] - trends
            return detrend#, masked


        files = glob(path+'lcs/*.csv')
        for file in files:
            # try:
            #db_ind = file.split('/')[-1].split('.csv')[0] + '.png' == exos['fig_names'].values
            #ref = exos['ref_counts'].iloc[db_ind].values
            lc = pd.read_csv(file)
            if np.nanmax(lc['event'].values) > 0:
                lc = np.array([lc['mjd'].values,lc['counts'].values])
                #lc[1] += ref
                lc = detrend_stellar_var(lc,sig_low=3,variable=True)
                lc2 = pd.read_csv(file)
                lc2['counts'] = lc[1]
                m,m, std = sigma_clipped_stats(lc[1])
                sig = (lc[1]-1)/std
                still_sig = np.nanmedian(sig[lc2['event'].values > 0]) < -3
                if still_sig:
                    lc2.to_csv(save_path+'norm_lcs/'+ file.split('/')[-1].split('.csv')[0] + '_norm_good.csv',index=False)
                    plt.figure()
                    # for i in range(int(np.nanmax(lc2['event'].values))-1):
                    #     i += 1
                    #     l = lc2.loc[lc2['event'] == i]
                    #     t = l['mjd'].values
                    #     plt.axvspan(t[0],t[-1],color='C1',alpha=0.6)
                    l = lc2.counts.values
                    l[l > 1.10] = np.nan
                    l[l < 0] = np.nan
                    plt.plot(lc2.mjd.values,lc2.counts.values)
                    
                    plt.ylabel('Normalized flux')
                    plt.xlabel('MJD')
                    plt.savefig(save_path+'norm_figs/'+ file.split('/')[-1].split('.csv')[0] + '_norm_good.png')
                else:
                    lc2.to_csv(save_path+'norm_lcs/'+ file.split('/')[-1].split('.csv')[0] + '_norm.csv',index=False)
            # except:
            #     pass


    def find_exos(self):
        eod_table = ExoplanetOrbitDatabase.get_table()
        exo_coords = []
        for i in range(0,len(eod_table)):
            exo_coords.append([float(eod_table['sky_coord'][i].to_string().split(' ')[0]), float(eod_table['sky_coord'][i].to_string().split(' ')[1])])
        crange = 0.005 # 20'' in degrees


        def get_num_dips(lc):
            dips = lc[:,0][np.argsort(lc[:,1])[:30]]
            # print('dips: ', dips)
            arr_sorted = np.sort(dips)
            result = []
            current_group = [arr_sorted[0]]
            for i in range(1, len(arr_sorted)):
                if abs(arr_sorted[i] - arr_sorted[i-1]) <= 1:
                    current_group.append(arr_sorted[i])
                else:
                    result.append(current_group)
                    current_group = [arr_sorted[i]]
            result.append(current_group)
            dp = [[] for i in range(0,len(result))]
            for i in range(0,len(dp)):
                for j in dips:
                    if j in result[i]:
                        dp[i].append(lc[:,1][np.where(lc[:,0]==j)][0])
            check_single_dip, max_dips = [], []
            for i in dp:
                check_single_dip.append(np.min(i))
                max_dips.append(lc[:,0][np.where(lc[:,1]==np.min(i))][0])
            gaps = []
            for n in range(0,len(max_dips)):
                for i in range(0,len(max_dips)):
                    if n != i:
                        gaps.append(np.abs(max_dips[i]-max_dips[n]))
            # print('dips: ', dips, 'arr_sorted: ', arr_sorted, 'result: ', result, 'max_dips: ', max_dips, 'gaps: ', gaps)
            arr_sorted = np.sort(gaps)
            result = []
            try:
                if np.sort(check_single_dip)[-1] - np.sort(check_single_dip)[-2] > 0.025:
                    result.append(1)
                    return len(result)
                current_group = [arr_sorted[0]]
                for i in range(1, len(arr_sorted)):
                    if abs(arr_sorted[i] - arr_sorted[i-1]) <= 0.1:
                        current_group.append(arr_sorted[i])
                    else:
                        result.append(current_group)
                        current_group = [arr_sorted[i]]
                result.append(current_group)
            except:
                result.append(1)
            return len(result)


        def deg_to_hms(ra,dec,units='h-d'):
            h, ms = int(ra/15), ra/15 - int(ra/15)
            m, s = int(ms*60), ms*60 - int(ms*60)
            s*= 60
            ra = f'{h}h{m}m{s:.5f}s'

            if units == 'h-d': 
                d, ms = int(dec), dec - int(dec)
                m, s = int(ms*60), ms*60 - int(ms*60)
                s*= 60
                dec = f'{d}d{abs(m)}m{abs(s):.5f}s'
            elif units == 'h-h':
                h, ms = int(dec/15), dec/15 - int(dec/15)
                m, s = int(ms*60), ms*60 - int(ms*60)
                s*= 60
                dec = f'{h}h{abs(m)}m{abs(s):.5f}s'

            return ra, dec


        path = f'{self.save_path}/Sector{self.sector}/'
        exo = pd.read_csv(path+'events_likely.csv')
        files = glob(path+'norm_lcs/*good.csv')
        for file in files:
            name = file.split('/')[-1].split('.')[0]
            known = False
            df = pd.read_csv(file)
            lc = df[['mjd','counts']].values
            #Visual lightcurve cuts
            _,cam,ccd,cut = file.split('/')[-1].split('C')
            cut,obj = cut.split('O')
            obj = obj.split('_')[0]
            ind = (exo['camera'].values==int(cam)) & (exo['ccd'].values==int(ccd)) & (exo['cut'].values==int(cut)) & (exo['objid'].values==int(obj))
            ref = exo.iloc[ind]
            if (np.min(lc[:,1]) < 0.92) | (get_num_dips(lc) < 2):
                df.to_csv(path+'potential_exo/possibles/lcs/'+file.split('/')[-1],index=False)
                call = f'cp {path}norm_figs/{name}.png {path}potential_exo/possibles/figs'
                os.system(call)
            else:
                coords = ref[['ra','dec']].values[0]
                ra, dec = coords[0], coords[1]
                try:
                    if len(ra) > 1:
                        ra = ra[0]
                        dec = dec[0]
                except:
                    pass
                #Astroquery
                # ra_, dec_ = deg_to_hms(ra,dec,units='h-d')
                # tab = Vizier.query_region(SkyCoord(ra_,dec_,frame='icrs'),radius=10*u.arcsec,catalog=['I/243/out'])#,'I/252/out','I/271/out','I/284/out'])
                #Save out
                for test in exo_coords:

                    if (test[0] > ra-crange) & (test[0] < ra+crange) & (abs(test[1]) > abs(dec)-crange) & (abs(test[1]) < abs(dec)+crange):
                        df.to_csv(path+'potential_exo/known_exos/lcs/'+file.split('/')[-1],index=False)
                        call = f'cp {path}norm_figs/{name}.png {path}potential_exo/known_exos/figs'
                        os.system(call)
                        known = True
                    else:
                        df.to_csv(path+'potential_exo/candidates/lcs/'+file.split('/')[-1],index=False)
                        call = f'cp {path}norm_figs/{name}.png {path}potential_exo/candidates/figs'
                        os.system(call)
                # I'm not sure what this is doing
                # if (ref[(ref.ra == ra) & (ref.dec == dec)].likely.values[0] != 'No') & (known == False):
                #     if len(tab) > 0:
                #         try:
                #             mag = tab[0][np.where(tab[0]['Rmag']<16.5)]
                #         except:
                #             mag = tab[0][np.where(tab[0]['Bmag']<16.5)]
                #         if len(mag) > 0:
                #             df.to_csv(path+'potential_exo/candidates/lcs/'+file.split('.')[0])
                #             name, end = file.split('.')
                #             call = f'cp {path}norm_figs/{name}.png {path}potential_exo/candidates/figs'
                #             os.system(call)
                #         elif len(tab[0]) > 0:
                #             df.to_csv(path+'potential_exo/possibles/lcs/'+file.split('.')[0])
                #             name, end = file.split('.')
                #             call = f'cp {path}norm_figs/{name}.png {path}possibles/figs'
                #             os.system(call)


    def main(self):
        if self.build_path == True:
            self.build_directory()
        print('Directory Built')
        #self.panoptes_test()
        print('Quality cuts made')
        #self.get_ref_counts()
        
        self.flatten_lcs()
        print('Lightcurves detrended and flattened')
        self.classify_transit_type()
        print('Dips fit')
        self.find_exos()
        print('Final cuts made')
        print('Complete')