from skimage import transform as trans
import glob, os, numpy, random
from tqdm import tqdm
import torch, sys, os, numpy, time, glob, cv2, random
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import numpy as np
import insightface, onnx
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import onnxruntime as ort
#from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import logging
#import gc, shutil

Log_file = 'visual_preprocess_align.log'
logging.basicConfig(filename=Log_file, level=logging.DEBUG)

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(224, 224))

# paths to dataset
# downloaded image directory
video_dir = '/misc/lu/bf_scratch/patx/VoxCeleb1_faces/unzippedIntervalFaces/data'
#video_dir = '/misc/scratch02/reco/Corpora/VoxCeleb_AV_March2023/voxceleb1/vox1_all_vid'
output_aligned_face_dir = '/misc/scratch11/Voxceleb1_Data/Voxceleb1_Data/Insightface_Aligned_Images'
#output_image_dir = '/misc/lu/bf_scratch/patx/rajasegp/Voxceleb1_Data/Images'
#frame_count_path = '/misc/lu/bf_scratch/patx/rajasegp/Voxceleb1_Data'
if not os.path.isdir(output_aligned_face_dir):
	os.makedirs(output_aligned_face_dir)
## Running parallel jobs
parser = argparse.ArgumentParser(description='Create Configuration')
parser.add_argument('--data_start_range', type=str, help='optional filename', 
	   default="")
parser.add_argument('--data_end_range', type=str, help='optional filename', 
	   default="")
args = parser.parse_args()
start_range = args.data_start_range
end_range = args.data_end_range

speaker_ids = os.listdir(video_dir)[int(start_range):int(end_range)]
logging.info("Number of Speakers: {0}".format(len(speaker_ids)))

#framecount_file_path = os.path.join(frame_count_path, "framecount.txt")
#framecount_file = open(framecount_file_path, "w")

for speaker in tqdm(speaker_ids, total=len(speaker_ids),position=0, leave=True):
	sub_dir_path = os.path.join(video_dir, speaker, '1.6')
	os.makedirs(os.path.join(output_aligned_face_dir,speaker), exist_ok = True)

	for speaker_dir in os.listdir(sub_dir_path):
		# Loading the speaker directory
		speaker_videos = os.path.join(sub_dir_path, speaker_dir)
		os.makedirs(os.path.join(output_aligned_face_dir,speaker,speaker_dir), exist_ok = True)

		for speaker_vid in os.listdir(speaker_videos):
			# Loading the speaker video
			speaker_video_path = os.path.join(speaker_videos, speaker_vid)
			os.makedirs(os.path.join(output_aligned_face_dir,speaker,speaker_dir, speaker_vid), exist_ok = True)

			for image in os.listdir(speaker_video_path):
				image_path = os.path.join(speaker_video_path, image)
				# Reading teh frame in the video
				img = cv2.imread(image_path)
				img = cv2.resize(img, (224,224))
				# Detecting and aligning the face region using Insight Face
				faces = app.get(img)
				# Checking the number of faces in the frame
				if len(faces) > 0:
					sizes = []
					for i in range(len(faces)):
						box = faces[i].bbox
						size = (box[3] - box[1]) * (box[2] - box[0])
						sizes.append(size)
					max_index = sizes.index(max(sizes))
					landmark = faces[max_index].kps
					tform = trans.SimilarityTransform()
					src = np.array([
								[30.2946, 51.6963],
								[65.5318, 51.5014],
								[48.0252, 71.7366],
								[33.5493, 92.3655],
								[62.7299, 92.2041]], dtype=np.float32)
					src[:, 0] += 8.0
					tform.estimate(landmark, src)
					M = tform.params[0:2, :]
					# Applying Affine transformation 
					img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
					save_path = os.path.join(output_aligned_face_dir, speaker, speaker_dir, speaker_vid, image)
					# saving the cropped and aligned faces
					if not os.path.isfile(save_path):
						cv2.imwrite(save_path, img)