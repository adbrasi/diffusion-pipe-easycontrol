import sys, numpy as np
from PIL import Image
o = sys.argv[1]; r = np.load(o + '_ref.npz'); c = np.load(o + '_comfy.npz')
rel = lambda a, b: float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)) / np.linalg.norm(b.reshape(-1)))
print('shapes ref/comfy final', r['final'].shape, c['final'].shape)
print('ref latent      rel diff %.4f' % rel(c['ref_latent'], r['ref_latent']))
print('step-0 denoised rel diff %.4f' % rel(c['den0'], r['den0']))
print('final latent    rel diff %.4f' % rel(c['final'], r['final']))
a = np.asarray(Image.open(o + '_ref.png')).astype(float); b = np.asarray(Image.open(o + '_comfy.png')).astype(float)
print('image PSNR %.2f dB' % (10 * np.log10(255 ** 2 / ((a - b) ** 2).mean())))
