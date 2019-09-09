# Classic CNN: AlexNet, GoogLeNet(+Inception_v3), VggNet
import cv2
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np
# from random import sample
from enum import Enum

os_env = 'MAC' #'WIN'
use_gpu = torch.cuda.is_available()
use_pth = False
save_cam = False
feat_visual = True

if os_env == 'MAC':
	save_img = '/Users/apple/Downloads/cnnKCF_proj_code/KCF_8.12/CNN_feat/img_save/'
	path_img = '/Users/apple/Downloads/BenchMark/dataset_5/train/Soccer/'
	save_feat= '/Users/apple/Downloads/cnnKCF_proj_code/KCF_8.12/CNN_feat/feat_save/'
	path_pth = '/Users/apple/.cache/torch/checkpoints/'
elif os_env == 'WIN':
	save_img = 'img_save\\'
	path_img = 'David\\'
	save_feat = 'feat_save\\'
	path_pth = None

class ClassicCNN(Enum):
	AlexNet, GoogLeNet, VggNet = 1,2,3

def draw_features(width, height, x, savename):
	tic=time.time()
	fig = plt.figure(figsize=(16, 16))
	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
	for i in range(width*height):
		plt.subplot(height,width, i + 1)
		plt.axis('off')
		# plt.tight_layout()
		img = x[0, i, :, :]
		pmin = np.min(img)
		pmax = np.max(img)
		img = (img - pmin) / (pmax - pmin + 0.000001)
		plt.imshow(img, cmap='gray')
		print("{}/{}".format(i,width*height))
	fig.savefig(savename, dpi=100)
	fig.clf()
	plt.close()
	print("time:{}".format(time.time()-tic))

preprocess = transforms.Compose([transforms.Resize((224,224)),
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)) ])


def returnCAM(feature_conv, weight_softmax, class_idx):
	# generate the class activation maps upsample to 256x256
	size_upsample = (224, 224)
	bz, nc, h, w = feature_conv.shape
	print('feature_conv.shape: ', feature_conv.shape)
	output_cam = []
	for idx in class_idx:
		print('weight_softmax[idx]: ', weight_softmax[idx].shape, weight_softmax[idx])
		cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
		cam = cam.reshape(h, w)
		cam = cam - np.min(cam)
		cam_img = cam / np.max(cam) # get CAM feature
		print('CAM feature: ', cam_img.shape)
		cam_img = np.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample))
	return output_cam


class Classic_feat(torch.nn.Module):
	def __init__(self, model_type=3, cnn_type=19, cnn_layer_test=0):
		super(Classic_feat, self).__init__()
		self.model_type = model_type
		self.cnn_type = cnn_type
		self.bn = True # BenchNorm
		self.use_layer = cnn_layer_test
		
		# select cnn_model from AlexNet, GoogLeNet, VggNet
		if self.model_type == ClassicCNN.AlexNet.value:
			print('use AlexNet')
			self.model = models.alexnet(pretrained=True)
			self.sel_feat_layer = {0,3,6,12} # conv1, conv2, conv3, conv5
		
		elif self.model_type == ClassicCNN.GoogLeNet.value:
			if self.cnn_type == 1: # GoogLeNet v1
				print('use GoogLeNet v1')
				self.model = models.googlenet(pretrained=True)
			else:
				print('use Inception v3')
				self.model = models.inception_v3(pretrained=True)
		
		elif self.model_type == ClassicCNN.VggNet.value:
			if cnn_type == 16:
				print('use Vgg16')
				self.model = models.vgg16(pretrained=True) # 31
				self.sel_feat_layer = {2,7,12,19,30} # conv1_1, conv2_2, conv3_2, conv4_2, conv5_4
			else: # use Vgg19 with bn
				self.cnn_type = 19
				if self.bn:
					print('use Vgg19_bn')
					self.model = models.vgg19_bn(pretrained=True)
					self.sel_feat_layer = {3,10,17,30,52} # conv1, conv2, conv3, conv4, conv5
				else:
					print('use Vgg19')
					self.model = models.vgg19(pretrained=True) # 37
					self.sel_feat_layer = {2,7,12,21,36} # conv1, conv2, conv3, conv4, conv5
		
		# use grad-CAM
		params = list(self.model.parameters())
		self.weight_softmax = np.squeeze(params[-2].data.numpy())
		self.CAMs = None
		
		self.model.eval()
		if use_gpu:
			print('use cuda model')
			self.model = self.model.cuda()
		if self.model_type != ClassicCNN.GoogLeNet.value:
			self.features = nn.ModuleList(list(self.model.features)).eval()
			self.classify = nn.ModuleList(list(self.model.classifier)).eval()

		# test feature
		if self.use_layer == 0: # run combine cnn_features
			self.use_layer = -4
			print('test linear combined cnn_features')
		elif self.use_layer > 4: # run combine features with grad-CAM
			print('test combined cnn_features with grad-CAM')
		else: # test each conv layers
			print('test single conv layer', self.use_layer)

	def forward(self, x):
		feat = []
		sel_layer = abs(self.use_layer)
		if self.model_type != ClassicCNN.GoogLeNet.value: # AlexNet, Vgg
			for ii, model in enumerate(self.features):
				x = model(x)
				# print(ii, "x: ", x.shape)
				if ii in self.sel_feat_layer: # get feat layer
					# print("get featture: ", ii, x.shape)
					feat.append(x.cpu().numpy())
					if (len(feat) == sel_layer):
						return x,feat
			# run classifier (GAP+FC)
			x = x.view(x.size(0), -1) # flaten to vector
			for ii, model in enumerate(self.classify):
				x = model(x)
				# print(ii, "x: ", x.shape)
		
		else: # GoogLeNet_v1, v3
			if self.cnn_type == 1: # GoogLeNet v1
				x = self.model.conv1(x) # 64*112*112 conv1_1
				feat.append(x.cpu().numpy()) # feat1 112
				x = self.model.maxpool1(x) # 64*56*56
				if self.use_layer == 1: return x,feat
				
				x = self.model.conv2(x)  # 64*56*56
				x = self.model.conv3(x)  # 192*56*56 conv2_2
				feat.append(x.cpu().numpy()) # feat2 56
				x = self.model.maxpool2(x)    # 192*28*28
				if self.use_layer == 2: return x,feat

				x = self.model.inception3a(x) # 256*28*28
				x = self.model.inception3b(x) # 480*28*28 conv3_2
				feat.append(x.cpu().numpy())  # feat3 28
				x = self.model.maxpool3(x)    # 480*14*14
				if self.use_layer == 3: return x,feat

				x = self.model.inception4a(x) # 512*14*14
				x = self.model.inception4b(x) # 512*14*14 conv4_2
				feat.append(x.cpu().numpy())  # feat4 14
				x = self.model.inception4c(x) # 512*14*14
				x = self.model.inception4d(x) # 528*14*14
				x = self.model.inception4e(x) # 832*14*14
				x = self.model.maxpool4(x)    # 832*7*7
				if self.use_layer == 4: return x,feat

				x = self.model.inception5a(x) # 832*7*7
				x = self.model.inception5b(x) # 1024*7*7 con5_2
				feat.append(x.cpu().numpy()) # feat5 7
				# GAP + FC
				x = self.model.avgpool(x) # 1024 * 1*1
				x = torch.flatten(x, 1)   # 1024
				x = self.model.dropout(x)
				x = self.model.fc(x) # 1000 (num_classes)
			else: # Inception_v3
				x = self.model.Conv2d_1a_3x3(x) # 32*111*111
				x = self.model.Conv2d_2a_3x3(x) # 32*109*109
				feat.append(x.cpu().numpy()) # feat1: conv1_2 109
				x = self.model.Conv2d_2b_3x3(x) # 64*109*109
				x = F.max_pool2d(x, kernel_size=3, stride=2)
				if self.use_layer == 1: return x,feat
				
				x = self.model.Conv2d_3b_1x1(x) # 80*54*54
				feat.append(x.cpu().numpy()) # feat2: conv2_1 54
				x = self.model.Conv2d_4a_3x3(x) # 192*52*52
				x = F.max_pool2d(x, kernel_size=3, stride=2)
				if self.use_layer == 2: return x,feat

				x = self.model.Mixed_5b(x) # 256*25*25
				x = self.model.Mixed_5c(x) # 288*25*25
				feat.append(x.cpu().numpy()) # feat3: conv3_2 25
				x = self.model.Mixed_5d(x) # 288*25*25
				if self.use_layer == 3: return x,feat
				
				x = self.model.Mixed_6a(x) # 768*12*12
				x = self.model.Mixed_6b(x) # 768*12*12
				feat.append(x.cpu().numpy()) # feat4: conv4_2
				x = self.model.Mixed_6c(x) # 768*12*12
				x = self.model.Mixed_6d(x) # 768*12*12
				x = self.model.Mixed_6e(x) # 768*12*12
				if self.use_layer == 4: return x,feat
				
				x = self.model.Mixed_7a(x) # 1280*5*5
				x = self.model.Mixed_7b(x) # 2048*5*5
				x = self.model.Mixed_7c(x) # 2048*5*5
				feat.append(x.cpu().numpy()) # feat5: CAM conv5_3
				# GAP + FC
				x = F.adaptive_avg_pool2d(x, (1, 1)) # 2048*1*1
				x = F.dropout(x, training=False) # 2048*1*1
				x = torch.flatten(x, 1) # 2048
				x = self.model.fc(x) # 1000 (num_classes)
		
		return x,feat


	def get_cnn_feat(self, img, feat_visual=False, use_CAM=False, save_cam=False):
		height, width = 224,224
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_tensor = preprocess(img)
		img_tensor = img_tensor.unsqueeze(0)
		if use_gpu:
			img_tensor = img_tensor.cuda()

		with torch.no_grad():
			out,feat = self.forward(img_tensor)
			if feat_visual and self.use_layer > 2:
				print('feat shape:', feat[0].shape, feat[1].shape, feat[2].shape)
				if self.model_type == ClassicCNN.AlexNet.value:
					draw_features(8, 8, feat[0], save_feat + 'Alex_feat1_56.png')
					draw_features(10,10,feat[1], save_feat + 'Alex_feat2_27.png')
				elif self.model_type == ClassicCNN.GoogLeNet.value:
					if self.cnn_type == 1:
						draw_features(8, 8, feat[0], save_feat + 'GoogLe_feat1_112.png')
						draw_features(10,10,feat[1], save_feat + 'GoogLe_feat2_56.png')
						draw_features(10,10,feat[2], save_feat + 'GoogLe_feat3_28.png')
					else:
						draw_features(5, 5, feat[0], save_feat + 'Inception_feat1_109.png')
						draw_features(8, 8, feat[1], save_feat + 'Inception_feat2_54.png')
						draw_features(10,10,feat[2], save_feat + 'Inception_feat3_25.png')
				elif self.model_type == ClassicCNN.VggNet.value:
					draw_features(8, 8, feat[0], save_feat + 'Vgg'+str(self.cnn_type)+'_feat_224.png')
					draw_features(8, 8, feat[1], save_feat + 'Vgg'+str(self.cnn_type)+'_feat_56.png')
					draw_features(10,10,feat[2], save_feat + 'Vgg'+str(self.cnn_type)+'_feat_28.png')

			# Alex, Vgg CAN'T use CAM
			if use_CAM:
				print('feat_cam: ', feat[-1].shape, 'logit out: ', out.shape)
				# [1, 512, 14, 14], [1, 512, 7, 7]
				# get class probs
				h_x = F.softmax(out, dim=1).data.squeeze()
				print('h_x: ', h_x.shape)
				probs, idx = h_x.sort(0, True)
				if use_gpu:
					probs, idx = probs.cpu(), idx.cpu()
				probs = probs.numpy()
				idx = idx.numpy()
				
				# run Grad_CAM with top 1 class
				self.CAMs = returnCAM(feat[-1], self.weight_softmax, [idx[0]])
				heatmap = cv2.applyColorMap(cv2.resize(self.CAMs[0],(width, height)), cv2.COLORMAP_JET)
				if save_cam:
					img.save(save_img + 'classic_test.jpg')
					img = cv2.imread(save_img + 'classic_test.jpg')
					result = heatmap * 0.3 + img * 0.5
					cv2.imwrite(save_img + 'classic_CAM.jpg', result)
					print('saved CAM_img')

			if self.use_layer <= 0: # linear combined cnn_features
				# use conv layer 1,3,4
				feat.pop(1)
				return feat
			elif self.use_layer > 4:
				# use conv layer 1,2,3 + grad-CAM
				feat.append(self.CAMs)
				return feat
			else: # test single feat layer
				# print(feat[0].shape, feat[0].dtype)
				return feat[self.use_layer-1]
