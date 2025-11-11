from exoplanet_hunter import ExoHunter
import matplotlib
matplotlib.use('Agg')
exo = ExoHunter(28,save_path = '/fred/oz335/projects/exoplanets/sorted',data_path='/fred/oz335/projects/exoplanets/sig5',run=False)
exo.build_directory()
exo.flatten_lcs()
exo.classify_transit_type()
exo.find_exos()