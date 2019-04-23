import os
import logging
import argparse
import numpy as np

from train_and_evaluate import evaluate, train
from model.net import Generator, Discriminator
from model.deep_net import ResGenerator, ResDiscriminator
from data_loader import fetch_dataloader
import utils
import torch
import scipy.io as io

restore_path = None
#restore_path = 'outputs_327_normal_noise0/model/model.pth.tar'

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',default='PG_GAN_417_high_res_test_8noise')

parser.add_argument('--train_path', default='TO_5k.nc',
										help="The training dataset path")
parser.add_argument('--json_output_dir',default='')

parser.add_argument('--train_dir',default='')

parser.add_argument('--restore_from', default=restore_path,
										help="Optional, directory or file containing weights to reload before training")
#parser.add_argument('--restore_from_his', default=restore_path_his)
parser.add_argument('--model', default='shallow',
										help="model to run")


if __name__ == '__main__':
	# Load the directory from commend line
	args = parser.parse_args()
	train_path = args.train_path
	output_dir = args.output_dir
	restore_from = args.restore_from

	#os.makedirs(output_dir + '/outputs', exist_ok = True)
	#os.makedirs(output_dir + '/figures', exist_ok = True)
	os.system("mkdir -p PG_GAN_417_high_res_test_8noise/efficiency")
	os.system("mkdir -p PG_GAN_417_high_res_test_8noise/results")
	os.system("mkdir -p PG_GAN_417_high_res_test_8noise/figures")
	os.system("mkdir -p PG_GAN_417_high_res_test_8noise/model")
        

	 # Set the logger
	utils.set_logger(os.path.join(args.train_dir, 'train.log'))


	# Load parameters from json file
	json_path = os.path.join('Params.json')
	assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
	params = utils.Params(json_path)

	# Add attributes to params
	params.output_dir = output_dir
	params.lambda_gp  = 10.0
	params.n_critic = 1
	params.cuda = torch.cuda.is_available()


	params.B = 100
	params.N = [0.5932, 0.5, 0.4068]
#	params.N = [0.5,0.5,0.5]
	params.R = 60
	params.beam_blur = 30

	
	params.batch_size = int(params.batch_size)
	params.numIter = int(params.numIter)
	params.noise_dims = int(params.noise_dims)
	params.label_dims = int(params.label_dims)
	params.gkernlen = int(params.gkernlen)
	params.step_size = int(params.step_size)
	params.iters_step,params.save_step, params.iters, params.noise_level= 100,100,0,0.1
	params.iters_scheme = [0,0,0,1]
	#params.iters_scheme = [0,0,0,1]
	#params.iters_scheme = [0,0,0,1]
	params.max_res = len(params.iters_scheme)
	params.size_temp = 64
	# fetch dataloader
	dataloader,gen_loss_list,dis_loss_list = fetch_dataloader(train_path, params),[],[]

	# Define the models 
	if args.model == 'shallow':
		generator = Generator(params)
		discriminator = Discriminator(params)
	elif args.model == 'deep':
		generator = ResGenerator(params)
		discriminator = ResDiscriminator(params)


	if params.cuda:
		generator.cuda()
		discriminator.cuda()


	# Define the optimizers 
	optimizer_G, params.gen_loss_list = torch.optim.Adam(generator.parameters(), lr=params.lr_gen, betas=(params.beta1_gen, params.beta2_gen)),[]
	optimizer_D, params.dis_loss_list = torch.optim.Adam(discriminator.parameters(), lr=params.lr_dis, betas=(params.beta1_dis, params.beta2_dis)),[]
	params.dis_loss_real_list, params.dis_loss_fake_list= [],[]
   
        #params.gen_loss_list=gen_loss_list
        #params.dis_loss_list=dis_loss_list

	if restore_from is not None:
		params.iters,generator, discriminator, gen_loss_list1 , dis_loss_list1, dis_loss_real_list1, dis_loss_fake_list1= utils.load_checkpoint(restore_from, (generator, discriminator), (optimizer_G, optimizer_D))
		for loss_index in range(len(gen_loss_list1)):
		    params.gen_loss_list.append(gen_loss_list1[loss_index].cpu().numpy())
		    params.dis_loss_list.append(dis_loss_list1[loss_index].cpu().numpy())
		    params.dis_loss_real_list.append(dis_loss_real_list1[loss_index].cpu().numpy())
		    params.dis_loss_fake_list.append(dis_loss_fake_list1[loss_index].cpu().numpy())   
        		
	# Define the schedulers
	scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=params.step_size, gamma = params.gamma)
	scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=params.step_size, gamma = params.gamma)


	# train the model and save 
	logging.info('Start training')
	loss_history = train((generator, discriminator), (optimizer_G, optimizer_D), (scheduler_G, scheduler_D), dataloader, params)
       # gen_loss_list = loss_history[0]
       # dis_loss_list = loss_history[1]
	# plot loss history and save
	#utils.plot_loss_history(loss_history, output_dir)
       # gen_loss_list=[]
       # dis_loss_list=[]
       #	for loss_index in range(len(loss_history[0])):
#	    gen_loss_list.append(loss_history[0][loss_index].cpu().numpy())
#	    dis_loss_list.append(loss_history[1][loss_index].cpu().numpy())
	# plot loss history and save
	#utils.plot_loss_history(loss_history, output_dir)
	filename = 'gen_loss_data'+'.mat'
	file_path_gen = os.path.join(params.output_dir,filename)

	io.savemat(file_path_gen,mdict= {'loss':params.gen_loss_list})  
	filename = 'dis_loss_data'+'.mat'

	file_path_dis = os.path.join(params.output_dir,filename)
	io.savemat(file_path_dis,mdict= {'loss':params.dis_loss_list})



	# Generate images and save 



	wavelengths = [w for w in range(500, 1301, 50)]
	angles = [a for a in range(35, 86, 5)]

	logging.info('Start generating devices for wavelength range {} to {} and angle range from {} to {} \n'
				.format(min(wavelengths), max(wavelengths), min(angles), max(angles)))
	evaluate(generator, wavelengths, angles, num_imgs=500, params=params)




