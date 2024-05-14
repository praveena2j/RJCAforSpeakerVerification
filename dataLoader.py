import glob, numpy, os, random, soundfile, torch, cv2, wave
from scipy import signal
import torchvision.transforms as transforms
import sys 
import pandas as pd
import math 

def init_loader(args):
	trainloader = train_loader(**vars(args))
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	evalLoader = eval_loader(**vars(args))
	args.evalLoader = torch.utils.data.DataLoader(evalLoader, batch_size = 1, shuffle = False, num_workers = args.n_cpu, drop_last = False)
	return args

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, frame_len, **kwargs):
		self.train_path = train_path
		self.frame_len = frame_len * 160 + 240
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		self.data_list = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))		
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = line.split()[1]
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

		mapping_file = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/meta_info/voxceleb1_meta/vox1_meta.csv"
		self.audio_path = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1/"
		vid_data = pd.read_csv(mapping_file, header=None)
		video_data = vid_data.to_dict("list")[0][1:]
		self.ids_to_names = {}
		for item in video_data:
			item = item.split('\t')
			item_id = item[0]
			item_name = item[1]
			self.ids_to_names[item_name] =  item_id

	def __getitem__(self, index):
		file = self.data_list[index]
		label = self.data_label[index]
		segments = self.load_wav(file = file)
		#segments = torch.FloatTensor(numpy.array(segments))
		faces    = self.load_face(file = file)
		faces = torch.FloatTensor(numpy.array(faces)).squeeze(1)
		return segments, faces, label

	def load_wav(self, file):
		comps = os.path.normpath(file).split(os.sep)
		#audiofile = os.path.join(self.audio_path,self.ids_to_names[comps[0]],comps[2],comps[3].zfill(9))
		utterance, sr = soundfile.read(os.path.join(self.audio_path,self.ids_to_names[comps[0]],comps[2],comps[3].zfill(9)))
		if utterance.shape[0] <= (self.frame_len*8):
			_utterance = numpy.zeros(self.frame_len*8)
			_utterance[-utterance.shape[0]:] = utterance
			utterance = _utterance

		framelen = int(math.floor(len(utterance) / 8))
		#if utterance.shape[0] <= (self.frame_len*8):
		#	shortage = self.frame_len - utterance.shape[0]
		#	utterance = numpy.pad(utterance, (0, shortage), 'wrap')
		#startframe = random.choice(range(0, utterance.shape[0] - (self.frame_len)))

		audio_segments = []
		for i in range(8):
			if framelen <= self.frame_len:
			#if utterance.shape[0] <= (self.frame_len*8):
				segment = numpy.expand_dims(numpy.array(utterance[int(i*self.frame_len):int((i*self.frame_len)+self.frame_len)]), axis = 0)
			else:
				audio_segment = numpy.array(utterance[int(i*framelen):int((i*framelen)+framelen)])
				startframe = random.choice(range(0, audio_segment.shape[0] - (self.frame_len)))
				segment = numpy.expand_dims(numpy.array(audio_segment[int(startframe):int(startframe)+self.frame_len]), axis = 0)
			augtype = random.randint(0,4)
			if augtype == 0:   # Original
				segment = segment
			elif augtype == 1:
				segment = self.add_rev(segment, length = self.frame_len)
			elif augtype == 2:
				segment = self.add_noise(segment, 'speech', length = self.frame_len)
			elif augtype == 3: 
				segment = self.add_noise(segment, 'music', length = self.frame_len)
			elif augtype == 4:
				segment = self.add_noise(segment, 'noise', length = self.frame_len)
			audio_segments.append(segment[0])
		#audio_seq_segments = torch.from_numpy(numpy.array(audio_segments))
		audio_seq_segments = torch.FloatTensor(numpy.array(audio_segments))
		#return segment[0]
		return audio_seq_segments

	def load_face(self, file):
		comps1 = os.path.normpath(file.split(" ")[0]).split(os.sep)
		frames = glob.glob("%s/*.jpg"%(os.path.join(self.train_path, os.path.join(comps1[0], comps1[2], comps1[3][:-4]))))
		#frames = glob.glob("%s/*.jpg"%(os.path.join(self.train_path, 'frame_align', file[:-4])))
		#frames = glob.glob("%s/*.jpg"%(os.path.join(self.train_path, file[:-4])))
		face_images = numpy.zeros((32, 3, 112, 112), dtype=numpy.uint8)
		face_frames = []
		#num_frames = math.ceil(len(frames) / 4)
		#if num_frames < 8:
		#frame = random.choice(frames)
		images = []
		for frame in frames:
			frame = cv2.imread(frame)			
			face = cv2.resize(frame, (112, 112))
			face = numpy.array(self.face_aug(face))		
			face = numpy.transpose(face, (2, 0, 1))
			images.append(face)
		images = numpy.array(images)

		if images.shape[0] <= 32:
			face_images[-images.shape[0]:,:,:,:] = images
			images = face_images

		for i in range(8):
			if images.shape[0] <= 32:
				random_index = random.sample(range(i*4, (i*4)+4), 1)
			else:
				win_length = int(math.floor(images.shape[0] / 8))
				random_index = random.sample(range(i*win_length, (i*win_length)+win_length), 1)
			face_frames.append(images[random_index, :, :, :])
		#face = numpy.array(self.face_aug(face))		
		#face = numpy.transpose(face, (2, 0, 1))
		return face_frames

	def __len__(self):
		return len(self.data_list)

	def face_aug(self, face):		
		global_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.GaussianBlur(kernel_size=(5, 9),sigma=(0.1, 5)),
			transforms.RandomGrayscale(p=0.2)
		])
		return global_transform(face)

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes()
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length))
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio

class eval_loader(object):
	def __init__(self, eval_data, eval_list, eval_path, num_eval_frames = 5, **kwargs):        
		self.data_list, self.data_length = [], []
		self.eval_path = eval_path
		self.eval_data = eval_data
		self.frame_len = 100 * 160 + 240
		self.num_eval_frames = num_eval_frames
		lines = open(eval_list).read().splitlines()
		for line in lines:
			data = line.split()
			self.data_list.append(data[-2])
			self.data_length.append(float(data[-1]))

		inds = numpy.array(self.data_length).argsort()
		self.data_list, self.data_length = numpy.array(self.data_list)[inds], \
										   numpy.array(self.data_length)[inds]
		self.minibatch = []
		start = 0
		while True:
			frame_length = self.data_length[start]
			minibatch_size = max(1, int(100 // frame_length)) 
			end = min(len(self.data_list), start + minibatch_size)
			self.minibatch.append([self.data_list[start:end], frame_length])
			if end == len(self.data_list):
				break
			start = end

		mapping_file = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/meta_info/voxceleb1_meta/vox1_meta.csv"
		self.audio_path = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1/"
		vid_data = pd.read_csv(mapping_file, header=None)
		video_data = vid_data.to_dict("list")[0][1:]
		self.ids_to_names = {}
		self.names_to_ids = {}
		for item in video_data:
			item = item.split('\t')
			item_id = item[0]
			item_name = item[1]
			self.ids_to_names[item_name] =  item_id
			self.names_to_ids[item_id] =  item_name

	def __getitem__(self, index):
		data_lists, frame_length = self.minibatch[index]
		filenames, segments, faces = [], [], []

		for num in range(len(data_lists)):
			file_name = data_lists[num]
			filenames.append(file_name)

			comps1 = os.path.normpath(file_name).split(os.sep)
			
			if self.eval_data == 'test':
				utterance, sr = soundfile.read(os.path.join(self.audio_path,comps1[0],comps1[1],comps1[2]))
			else:
				utterance, sr = soundfile.read(os.path.join(self.audio_path,self.ids_to_names[comps1[0]],comps1[2],comps1[3].zfill(9)))

			if len(utterance) <= (self.frame_len*8):
				_utterance = numpy.zeros(self.frame_len*8)
				_utterance[-utterance.shape[0]:] = utterance
				utterance = _utterance

			framelen = int(math.floor(len(utterance) / 8))

			audio_segments = []
			for i in range(8):
				#if len(utterance) <= (self.frame_len*8):
				if framelen <= self.frame_len:
					segment = numpy.array(utterance[int(i*self.frame_len):int((i*self.frame_len)+self.frame_len)])
				else:
					audio_segment = numpy.array(utterance[int(i*framelen):int((i*framelen)+framelen)])
					startframe = random.choice(range(0, audio_segment.shape[0] - (self.frame_len)))
					segment = numpy.array(audio_segment[int(startframe):int((startframe)+self.frame_len)])

				audio_segments.append(segment)
			segments.append(numpy.array(audio_segments))

			#frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path, 'frame_align', file_name[:-4])))
			#frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path, file_name[:-4])))				
	

			comps1 = os.path.normpath(file_name.split(" ")[0]).split(os.sep)
			

			if self.eval_data == 'test':
				frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path,self.names_to_ids[comps1[0]], comps1[1], comps1[2][:-4].lstrip('0'))))
			else:
				frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path,comps1[0], comps1[2], comps1[3][:-4])))

			face_images = numpy.zeros((32, 3, 112, 112), dtype=numpy.uint8)
			face_frames = []
			#num_frames = math.ceil(len(frames) / 4)
			#if num_frames < 8:
			#frame = random.choice(frames)
			images = []
			for frame in frames:
				frame = cv2.imread(frame)			
				face = cv2.resize(frame, (112, 112))
				face = numpy.transpose(face, (2, 0, 1))
				images.append(face)
			images = numpy.array(images)

			if images.shape[0] <= 32:
				face_images[-images.shape[0]:,:,:,:] = images
				images = face_images

			for i in range(8):
				if images.shape[0] <= 32:
					random_index = random.sample(range(i*4, (i*4)+4), 1)
				else:
					win_length = int(math.floor(images.shape[0] / 8))
					random_index = random.sample(range(i*win_length, (i*win_length)+win_length), 1)
				face_frames.append(images[random_index, :, :, :])
			faces.append(numpy.array(face_frames))
		segments = torch.FloatTensor(numpy.array(segments))
		faces = torch.FloatTensor(numpy.array(faces))
		faces = faces.div_(255).sub_(0.5).div_(0.5)
		return segments, faces, filenames

	def __len__(self):
		return len(self.minibatch)