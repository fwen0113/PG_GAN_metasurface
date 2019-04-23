import os
import logging
from tqdm import tqdm

from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F 
import torch
import utils
import scipy.io as io


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def progress_output(p,p_list):
    output_index = 0
    for a_index in range(len(p_list)):

        if(p > p_list[a_index]):
            output_index += 1
    if(p_list[output_index] == 0):
        progress_alpha = 1
    else:
        p_range = p_list[output_index] - p_list[output_index-1]
        if(output_index == 0):
            progress_alpha = (p/p_list[output_index])
        else:   
            progress_alpha = (p-p_list[output_index-1])/p_range
        
    return output_index, progress_alpha



def visualize_training_generator(generator, fig_path,params,cuda, n_row = 4, n_col = 4):
    generator.eval()
    wavelengths = torch.linspace(-1, 1, n_col).view(1, n_col).repeat(n_row, 1).view(-1, 1)
    angles = torch.linspace(-1, 1, n_row).view(n_row, 1).repeat(1, n_col).view(-1, 1)
    cond = torch.cat([wavelengths, angles], -1).type(Tensor)
    imgs, _ = sample_images(generator, cond, params, cuda)
    paddings = (0, 0, 0, imgs.size(2)-1)
    imgs = F.pad(imgs, paddings, mode='reflect')
    save_image(imgs, fig_path, n_row)
    generator.train()


def sample_images(generator, cond, params,cuda):   
    if cuda:
        z = Variable(torch.cuda.FloatTensor(cond.size(0), generator.noise_dim).normal_())
        z.cuda()
    else:
        z = Variable(torch.randn(cond.size(0), generator.noise_dim))        
    return generator(z,cond,params), z


def evaluate(generator, wavelengths, angles, num_imgs, params):
    generator.eval()
    for wavelength in wavelengths:
        for angle in angles:
            filename = 'ccGAN_imgs_Si_w' + str(wavelength) +'_' + str(angle) +'deg.mat'
            mdict = {'wavelength': wavelength, 'angle': angle}

            w = (wavelength - params.wc)/params.wspan
            theta = (angle - params.ac)/params.aspan

            cond = Tensor([w, theta]).repeat(num_imgs, 1)
            images, noise = sample_images(generator, cond, params, params.cuda)

            mdict['imgs'] = torch.squeeze(images).cpu().detach().numpy()
            mdict['noise'] = noise.data.cpu().numpy()
            mdict['params'] = params
            mdict['wspan'] = params.wspan.cpu().detach().numpy()
            mdict['wc'] = params.wc.cpu().detach().numpy()
                   
            file_path = os.path.join(params.output_dir,'results',filename)
            io.savemat(file_path, mdict=mdict)

        logging.info('wavelength = '+str(wavelength)+ ' is done. \n')

def efficiency_save(generator, cuda, n_row, n_col,it,params):
    generator.eval()
    wavelengths = torch.linspace(-1, 1, n_col).view(1, n_col).repeat(n_row, 1).view(-1, 1)
    angles = torch.linspace(-1, 1, n_row).view(n_row, 1).repeat(1, n_col).view(-1, 1)
    wavelengths_new = wavelengths.clone() * params.wspan + params.wc
    angles_new = angles.clone() * params.aspan + params.ac  
    labels = torch.cat([wavelengths, angles], -1).type(Tensor)
    imgs, _ = sample_images(generator, labels, params,cuda)
    labels = torch.cat([wavelengths_new, angles_new], -1).type(Tensor)
    filename = 'img_at_'+ str(params.iters+it) +'.mat'
    file_path = os.path.join(params.output_dir,'efficiency',filename)
    temp = torch.squeeze(imgs).cpu().detach().numpy()
    mdict = {'imgs': temp,'it':params.iters+it}

    #mdict['dr_nm'] = dr_nm.detach().numpy()
    mdict['imgs'] = torch.squeeze(imgs).cpu().detach().numpy()
    mdict['params'] = params
    mdict['label']= labels.cpu().detach().numpy()
    io.savemat(file_path, mdict=mdict)
    generator.train() 


def compute_gradient_penalty(params,D, real_samples, fake_samples, cond, cuda=False):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).type(Tensor)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).type(Tensor).requires_grad_(True)
    d_interpolates = D(interpolates, cond,params)

    fake = Variable(Tensor(real_samples.size(0), 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def brute_norm(inputs):
    inputs_max = torch.max(inputs)
    inputs_min = torch.min(inputs)
    inputs = -1+(inputs - inputs_min)*2/(inputs_max - inputs_min)
    inputs =inputs.type(Tensor)
    return inputs

def train(models, optimizers, schedulers, dataloader, params):

    generator, discriminator = models
    optimizer_G, optimizer_D = optimizers
    scheduler_G, scheduler_D = schedulers


    generator.train()
    discriminator.train()

    gen_loss_history = []
    dis_loss_history = []
    dis_loss_real_history = []
    dis_loss_fake_history = []

    with tqdm(total=params.numIter) as t:
        it = 0
        while True:
            for i, (real_imgs, cond) in enumerate(dataloader):
                #print('cond size',cond.size())
                it +=1 
                

                scheduler_G.step()
                scheduler_D.step()
                    
                if it > params.numIter:
                              
		   # torch.save(torch.cat(gen_loss_history,dis_loss_history,-1),params.restore_path_loss)
                    return (gen_loss_history, dis_loss_history)

                params.progress = it/params.numIter
                print('i is',it)
                params.scheme_index, params.alpha = progress_output(params.progress,params.iters_scheme)

                # move to GPU if available
                if params.cuda:
                    real_imgs, cond = real_imgs.cuda(), cond.cuda()
          
                # convert to torch Variables
                Tensor = torch.cuda.FloatTensor if params.cuda else torch.FloatTensor
                real_imgs, cond = Variable(real_imgs.type(Tensor)), Variable(cond.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------
                

                optimizer_D.zero_grad()
                # Sample noise as generator input
                z = Variable(torch.randn(cond.size(0), params.noise_dims).type(Tensor))
                #if params.cuda:
                 #   z.cuda()
                # Generate a batch of images

                fake_imgs = generator(z,cond,params)

             
                # Real images

                real_imgs = F.adaptive_avg_pool2d(real_imgs,fake_imgs[0][0].size())


                real_validity = discriminator(real_imgs, cond,params)

             
                # Fake images
                
                fake_validity = discriminator(fake_imgs, cond,params)


                gradient_penalty = compute_gradient_penalty(params,discriminator, real_imgs.data, fake_imgs.data, cond.data, params.cuda)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + params.lambda_gp * gradient_penalty
                d_loss_real = torch.mean(real_validity)
                d_loss_fake = torch.mean(fake_validity)
                d_loss.backward()
                optimizer_D.step()

                dis_loss_history.append(d_loss.data)
                dis_loss_real_history.append(d_loss_real.data)
                dis_loss_fake_history.append(d_loss_fake.data)
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                 # Train the generator every n_critic steps
                if it % params.n_critic == 0:
                    # Generate a batch of images
                    fake_imgs = generator(z,cond,params)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs, cond,params)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    gen_loss_history += [g_loss.data] * params.n_critic
                    
                #t.set_postfix(loss='{:05.3f}'.format(g_loss.data))
                #t.update()
                
                
                if it % params.iters_step == 0:
                    fig_path = os.path.join(params.output_dir, 'figures', 'iter{}.png'.format(it+params.iters))
                    visualize_training_generator(generator, fig_path, params,params.cuda)
                
                n_row=4
                n_col=4    

                if it % params.save_step == 0:
                    efficiency_save(generator, params.cuda, n_row,n_col,it,params)

                    for loss_index in range(it-params.save_step,it):
                        params.gen_loss_list.append(gen_loss_history[loss_index].cpu().numpy())
                        params.dis_loss_list.append(dis_loss_history[loss_index].cpu().numpy())
                        params.dis_loss_real_list.append(dis_loss_real_history[loss_index].cpu().numpy())
                        params.dis_loss_fake_list.append(dis_loss_fake_history[loss_index].cpu().numpy())
                        filename = 'gen_loss_data'+'.mat'
                        file_path_gen = os.path.join(params.output_dir,filename)
                        io.savemat(file_path_gen,mdict= {'loss':params.gen_loss_list})
                        filename = 'dis_loss_data'+'.mat'
                        file_path_dis = os.path.join(params.output_dir,filename)
                        model_dir = os.path.join(params.output_dir, 'model')
                        io.savemat(file_path_dis,mdict= {'loss':params.dis_loss_list})
                        filename = 'dis_loss_real_data'+'.mat'
                        file_path_dis = os.path.join(params.output_dir,filename)
                        model_dir = os.path.join(params.output_dir, 'model')
                        io.savemat(file_path_dis,mdict= {'loss':params.dis_loss_real_list})
                        filename = 'dis_loss_fake_data'+'.mat'
                        file_path_dis = os.path.join(params.output_dir,filename)
                        model_dir = os.path.join(params.output_dir, 'model')
                        io.savemat(file_path_dis,mdict= {'loss':params.dis_loss_fake_list})



                        utils.save_checkpoint({'iters': params.iters+it,
                                           'gen_loss_history': gen_loss_history,
    					   'dis_loss_history': dis_loss_history,
                                           'dis_loss_real_history': dis_loss_real_history,
                                           'dis_loss_fake_history': dis_loss_fake_history,
                                           'gen_state_dict': generator.state_dict(),
                                           'dis_state_dict': discriminator.state_dict(),
                                           'optim_G_state_dict': optimizer_G.state_dict(),
                                           'optim_D_state_dict': optimizer_D.state_dict(),
                                           'scheduler_G_state_dict': scheduler_G.state_dict(),
                                           'scheduler_D_state_dict': scheduler_D.state_dict()},
                                            checkpoint=model_dir)

                t.set_postfix(loss='{:05.3f}'.format(g_loss.data))
                t.update()



