import numpy as np
import cv2
import scipy.io as sio
from modelo_rs import resp

def gaussian_kernel(dimension_x, dimension_y, sigma_x, sigma_y):
    x = cv2.getGaussianKernel(dimension_x, sigma_x)
    y = cv2.getGaussianKernel(dimension_y, sigma_y)
    kernel = y.dot(x.T)
    return kernel


def normalize_map(s_map):
	# normalize the salience map
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map

def discretize_gt(gt):
	return gt/255

def auc_judd(s_map,gt):
	# ground truth is discrete, s_map is continous and normalized
	gt = discretize_gt(gt)
	s_map = normalize_map(s_map)
	# thresholds are calculated from the salience map, only at places where fixations are present
	thresholds = []
	for i in range(0,gt.shape[0]):
		for k in range(0,gt.shape[1]):
			if gt[i][k]>0:
				thresholds.append(s_map[i][k])

	num_fixations = np.sum(gt)
	# num fixations is no. of salience map values at gt >0
	thresholds = sorted(set(thresholds))
	#fp_list = []
	#tp_list = []
	area = []
	area.append((0.0,0.0))
	for thresh in thresholds:
		# in the salience map, keep only those pixels with values above threshold
		temp = np.zeros(s_map.shape)
		temp[s_map>=thresh] = 1.0
		assert np.max(gt)==1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
		assert np.max(s_map)==1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
		num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
		tp = num_overlap/(num_fixations*1.0)
		
		# total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
		# this becomes nan when gt is full of fixations..this won't happen
		fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
		
		area.append((round(tp,4),round(fp,4)))
		#tp_list.append(tp)
		#fp_list.append(fp)

	#tp_list.reverse()
	#fp_list.reverse()
	area.append((1.0,1.0))
	#tp_list.append(1.0)
	#fp_list.append(1.0)
	#print tp_list
	area.sort(key = lambda x:x[0])
	tp_list =  [x[0] for x in area]
	fp_list =  [x[1] for x in area]
	return np.trapz(np.array(tp_list),np.array(fp_list))


def auc_shuff(s_map,gt,other_map,splits=100,stepsize=0.1):
	gt = discretize_gt(gt)
	other_map = discretize_gt(other_map)
	s_map = normalize_map(s_map)

	num_fixations = np.sum(gt)
	
	x,y = np.where(other_map==1)
	other_map_fixs = []
	for j in zip(x,y):
		other_map_fixs.append(j[0]*other_map.shape[0] + j[1])
	ind = len(other_map_fixs)
	assert ind==np.sum(other_map), 'something is wrong in auc shuffle'


	num_fixations_other = min(ind,num_fixations)

	num_pixels = s_map.shape[0]*s_map.shape[1]
	random_numbers = []
	for i in range(0,splits):
		temp_list = []
		t1 = np.random.permutation(ind)
		for k in t1:
			temp_list.append(other_map_fixs[k])
		random_numbers.append(temp_list)	

	aucs = []
	# for each split, calculate auc
	for i in random_numbers:
		r_sal_map = []
		for k in i:
			#print('{}  {}'.format(k%s_map.shape[0]-1, int(np.floor(k/s_map.shape[0]))))
			r_sal_map.append(s_map[k%s_map.shape[0]-1, int(np.floor(k/s_map.shape[0]))])
		# in these values, we need to find thresholds and calculate auc
		thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

		r_sal_map = np.array(r_sal_map)

		# once threshs are got
		thresholds = sorted(set(thresholds))
		area = []
		area.append((0.0,0.0))
		for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
			temp = np.zeros(s_map.shape)
			temp[s_map>=thresh] = 1.0
			num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
			tp = num_overlap/(num_fixations*1.0)
			
			#fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
			fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

			area.append((round(tp,4),round(fp,4)))
		
		area.append((1.0,1.0))
		area.sort(key = lambda x:x[0])
		tp_list =  [x[0] for x in area]
		fp_list =  [x[1] for x in area]

		aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))
	
	return np.mean(aucs)


def nss(s_map,gt):
	gt = discretize_gt(gt)
	s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)

	x,y = np.where(gt==1)
	temp = []
	for i in zip(x,y):
		temp.append(s_map_norm[i[0],i[1]])
	return np.mean(temp)


def cc(s_map,gt):
	s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
	gt_norm = (gt - np.mean(gt))/np.std(gt)
	a = s_map_norm
	b= gt_norm
	r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
	return r



if __name__ == '__main__':

	#Modifica isto segundo a tua estrutura de ficheiros
	path_to_origfixdata_mat = "../dataset_Toronto/Fixacions/" #path ata o ficheiro origfixdata.mat
	path_mapas_densidade = "../dataset_Toronto/MapasDensidade/" #path ao cartafol cos mapas de densidade
	path_to_imaxes = "../dataset_Toronto/Imaxes/"  #path ao directorio onde estan as imaxes da base de datos

	#Lemos a estrutra .mat das fixacions. Na variable fixacions poderemos
	#acceder asos datos como fixacions['white'][0,index] e devolveramos unha
	#imaxe onde nos pixeles que teñen fixacion teñen un valor de 1.
	#index debe variar entre 0 e numero maximo de imaxes na base de datos (120 en total)
	fixacions =  sio.loadmat(path_to_origfixdata_mat + 'origfixdata.mat')

	#imos ler todas as imaxe e visualiamos as tres 
	# opcions: fixacions, mapas de densidades e fixacion sobre a imaxe orixinal
	auc_l = []
	auc_s = []
	nss_l = []
	cc_l = []
	for index in range(len(fixacions['white'][0])):
		#Visualizamos as fixacions (*255 pq con valor 1 non se ven!)
		gt = fixacions['white'][0,index]*255
		#cv2.imshow('Mapa de Fixacions',gt)


		#Achamos o mapa de saliencia de algoritmo residuo espectral
		#print(path_to_imaxes + str(index+1) + ".jpg")
		s_map = resp(path_to_imaxes + str(index+1) + ".jpg")
		#s_map = gaussian_kernel(gt.shape[1],gt.shape[0], 80, 80)
		cv2.imshow('Saliency Map', cv2.normalize(s_map, None,0,1.0,cv2.NORM_MINMAX))

		#achamos as medidas
		auc_judd_score = auc_judd(s_map,gt)
		#print ('auc judd :', auc_judd_score)
		auc_l.append(auc_judd_score)

		#print('{}   {}'.format(len(fixacions['white'][0]), int(np.random.randint(len(fixacions['white'][0]),size=1))))
		gt_other = fixacions['white'][0,int(np.random.randint(len(fixacions['white'][0]),size=1))]*255
		#gt_other = gt.copy()
		cv2.imshow('Mapa de Fixacions Shuffle',gt_other)
		auc_shuffle_score = auc_shuff(s_map,gt, gt_other )
		#print ('auc shuffle :', auc_shuffle_score)
		auc_s.append(auc_shuffle_score)


		nss_score = nss(s_map,gt)
		#print('nss :', nss_score)
		nss_l.append(nss_score)

		#comparamos o mapa de saliencia co mapa de
		#densidade de fixación do humamo
		map_den = cv2.imread(path_mapas_densidade + "d" + str(index+1) + ".jpg",0)
		cv2.imshow('Densidade de fixacions', map_den)
		cc_score = cc(s_map,map_den)
		#print('cc :', cc_score)
		cc_l.append(cc_score)

		c = cv2.waitKey(1) & 0xFF
		if(c==27 or c==ord('q')):
			break
	print('nss_media={} auc_media={} auc_shuffle_media ={} cc_mean={}'.format(np.mean(nss_l), np.mean(auc_l), np.mean(auc_s), np.mean(cc_l)))
	cv2.destroyAllWindows()


