#include <sched.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include <alsa/asoundlib.h>
#include "alsa.h"

#include "partconvMulti.h"

#define GETCSR()    ({ int _result; __asm__ volatile ("stmxcsr %0" : "=m" (*&_result) ); /*return*/ _result; })
#define SETCSR( a )    { int _temp = a; __asm__ volatile( "ldmxcsr %0" : : "m" (*&_temp ) ); }

#define DISABLE_DENORMALS int _savemxcsr = GETCSR(); SETCSR(_savemxcsr | 0x8040);
#define RESTORE_DENORMALS SETCSR(_savemxcsr);

#define ROUNDMODE_NEG_INF int _savemxcsr = GETCSR(); SETCSR((_savemxcsr & ~0x6000) | 0x2000);
#define RESTORE_ROUNDMODE SETCSR(_savemxcsr);
#define SET_ROUNDMODE           ROUNDMODE_NEG_INF

#define kMaxFloat32 2147483520.0f

#ifdef __cplusplus
extern "C"
{
#endif

// ____________________________________________________________________________
//
// FloatToInt
// N.B. Functions which use this should invoke SET_ROUNDMODE / RESTORE_ROUNDMODE.
static inline int FloatToInt(double inf, double min32, double max32)
{
        if (inf >= max32) return 0x7FFFFFFF;
        return (int)inf;
}


void Float32ToNativeInt32( const float *src, int *dst, unsigned int numToConvert )
{
	const float *src0 = src;
	int *dst0 = dst;
	unsigned int count = numToConvert;
	
	if (count >= 4) {
		// vector -- requires 4+ samples
		ROUNDMODE_NEG_INF
		const __m128 vround = (const __m128) { 0.5f, 0.5f, 0.5f, 0.5f };
		const __m128 vmin = (const __m128) { -2147483648.0f, -2147483648.0f, -2147483648.0f, -2147483648.0f };
		const __m128 vmax = (const __m128) { kMaxFloat32, kMaxFloat32, kMaxFloat32, kMaxFloat32  };
		const __m128 vscale = (const __m128) { 2147483648.0f, 2147483648.0f, 2147483648.0f, 2147483648.0f  };
		__m128 vf0;
		__m128i vi0;
	
#define F32TOLE32(x) \
		vf##x = _mm_mul_ps(vf##x, vscale);			\
		vf##x = _mm_add_ps(vf##x, vround);			\
		vf##x = _mm_max_ps(vf##x, vmin);			\
		vf##x = _mm_min_ps(vf##x, vmax);			\
		vi##x = _mm_cvtps_epi32(vf##x);			\

		int falign = (uintptr_t)src & 0xF;
		int ialign = (uintptr_t)dst & 0xF;
	
		if (falign != 0 || ialign != 0) {
			// do one unaligned conversion
			vf0 = _mm_loadu_ps(src);
			F32TOLE32(0)
			_mm_storeu_si128((__m128i *)dst, vi0);
			
			// and advance such that the destination ints are aligned
			unsigned int n = (16 - ialign) / 4;
			src += n;
			dst += n;
			count -= n;

			falign = (uintptr_t)src & 0xF;
			if (falign != 0) {
				// unaligned loads, aligned stores
				while (count >= 4) {
					vf0 = _mm_loadu_ps(src);
					F32TOLE32(0)
					_mm_store_si128((__m128i *)dst, vi0);
					src += 4;
					dst += 4;
					count -= 4;
				}
				goto VectorCleanup;
			}
		}
	
		while (count >= 4) {
			vf0 = _mm_load_ps(src);
			F32TOLE32(0)
			_mm_store_si128((__m128i *)dst, vi0);
			
			src += 4;
			dst += 4;
			count -= 4;
		}
VectorCleanup:
		if (count > 0) {
			// unaligned cleanup -- just do one unaligned vector at the end
			src = src0 + numToConvert - 4;
			dst = dst0 + numToConvert - 4;
			vf0 = _mm_loadu_ps(src);
			F32TOLE32(0)
			_mm_storeu_si128((__m128i *)dst, vi0);
		}
		RESTORE_ROUNDMODE
		return;
	}
	
	// scalar for small numbers of samples
	if (count > 0) {
		double scale = 2147483648.0, round = 0.5, max32 = 2147483648.0 - 1.0 - 0.5, min32 = 0.;
		ROUNDMODE_NEG_INF
		
		while (count-- > 0) {
			double f0 = *src++;
			f0 = f0 * scale + round;
			int i0 = FloatToInt(f0, min32, max32);
			*dst++ = i0;
		}
		RESTORE_ROUNDMODE
	}
}


void NativeInt32ToFloat32( const int *src, float *dst, unsigned int numToConvert )
{
	const int *src0 = src;
	float *dst0 = dst;
	unsigned int count = numToConvert;

	if (count >= 4) {
		// vector -- requires 4+ samples
#define LEI32TOF32(x) \
	vf##x = _mm_cvtepi32_ps(vi##x); \
	vf##x = _mm_mul_ps(vf##x, vscale); \
		
		const __m128 vscale = (const __m128) { 1.0/2147483648.0f, 1.0/2147483648.0f, 1.0/2147483648.0f, 1.0/2147483648.0f  };
		__m128 vf0;
		__m128i vi0;

		int ialign = (uintptr_t)src & 0xF;
		int falign = (uintptr_t)dst & 0xF;
	
		if (falign != 0 || ialign != 0) {
			// do one unaligned conversion
			vi0 = _mm_loadu_si128((__m128i const *)src);
			LEI32TOF32(0)
			_mm_storeu_ps(dst, vf0);
			
			// and advance such that the destination floats are aligned
			unsigned int n = (16 - falign) / 4;
			src += n;
			dst += n;
			count -= n;

			ialign = (uintptr_t)src & 0xF;
			if (ialign != 0) {
				// unaligned loads, aligned stores
				while (count >= 4) {
					vi0 = _mm_loadu_si128((__m128i const *)src);
					LEI32TOF32(0)
					_mm_store_ps(dst, vf0);
					src += 4;
					dst += 4;
					count -= 4;
				}
				goto VectorCleanup;
			}
		}
	
		// aligned loads, aligned stores
		while (count >= 4) {
			vi0 = _mm_load_si128((__m128i const *)src);
			LEI32TOF32(0)
			_mm_store_ps(dst, vf0);
			src += 4;
			dst += 4;
			count -= 4;
		}
		
VectorCleanup:
		if (count > 0) {
			// unaligned cleanup -- just do one unaligned vector at the end
			src = src0 + numToConvert - 4;
			dst = dst0 + numToConvert - 4;
			vi0 = _mm_loadu_si128((__m128i const *)src);
			LEI32TOF32(0)
			_mm_storeu_ps(dst, vf0);
		}
		return;
	}
	// scalar for small numbers of samples
	if (count > 0) {
		double scale = 1./2147483648.0f;
		while (count-- > 0) {
			int i = *src++;
			double f = (double)i * scale;
			*dst++ = f;
		}
	}
}

int alsa_set_hwparams(alsa_dev_t *dev, snd_pcm_t *handle, snd_pcm_hw_params_t *params, snd_pcm_access_t access)
{
  unsigned int rrate;
  snd_pcm_uframes_t size;
  int err, dir;
  
  /* choose all parameters */
  err = snd_pcm_hw_params_any(handle, params);
  if (err < 0) {
    printf("Broken configuration for playback: no configurations available: %s\n", snd_strerror(err));
    return err;
  }
  
  /* set the interleaved read/write format */
  err = snd_pcm_hw_params_set_access(handle, params, access);
  if (err < 0) {
    printf("Access type not available for playback: %s\n", snd_strerror(err));
    return err;
  }
  /* set the sample format */
  err = snd_pcm_hw_params_set_format(handle, params, dev->format);
  if (err < 0) {
    printf("Sample format not available for playback: %s\n", snd_strerror(err));
    return err;
  }
  /* set the count of channels */
  err = snd_pcm_hw_params_set_channels(handle, params, dev->channels);
  if (err < 0) {
    printf("Channels count (%d) not available for playbacks: %s\n", dev->channels, snd_strerror(err));
    return err;
  }
  /* set the stream rate */
  rrate = dev->rate;
  err = snd_pcm_hw_params_set_rate_near(handle, params, &rrate, 0);
  if (err < 0) {
    printf("Rate %d Hz not available for playback: %s\n", dev->rate, snd_strerror(err));
    return err;
  }
  if (rrate != dev->rate) {
    printf("Rate doesn't match (requested %dHz, get %dHz)\n", dev->rate, rrate);
    return -EINVAL;
  }
  
  /* set the period size */
  err = snd_pcm_hw_params_set_period_size(handle, params, dev->period_size, 0);
  if (err < 0) {
    printf("Unable to set period size %d for playback: %s\n", (int)dev->period_size, snd_strerror(err));
    return err;
  }
  
  err = snd_pcm_hw_params_get_period_size(params, &size, &dir);
  if (err < 0) {
    printf("Unable to get period size for playback: %s\n", snd_strerror(err));
    return err;
  }
  
  if (dev->period_size != size) {
    printf("Period size doesn't match (requested %d, got %d)\n", (int)dev->period_size, (int)size);
    return -EINVAL;
  }
  
    /* set the buffer size */
  err = snd_pcm_hw_params_set_buffer_size(handle, params, dev->buffer_size);
  if (err < 0) {
    printf("Unable to set buffer size %d for playback: %s\n", (int)dev->buffer_size, snd_strerror(err));
    return err;
  }
  err = snd_pcm_hw_params_get_buffer_size(params, &size);
  if (err < 0) {
    printf("Unable to get buffer size for playback: %s\n", snd_strerror(err));
    return err;
  }
  
  if (size != (snd_pcm_uframes_t)dev->buffer_size) {
    printf("Buffer size doesn't match (requested %d, got %d)\n", (int)dev->buffer_size, (int)size);
    return -EINVAL;
  }

  /* write the parameters to device */
  err = snd_pcm_hw_params(handle, params);
  if (err < 0) {
    printf("Unable to set hw params for playback: %s\n", snd_strerror(err));
    return err;
  }
  return 0;
}

int alsa_set_swparams(alsa_dev_t *dev, snd_pcm_t *handle, snd_pcm_sw_params_t *swparams)
{
  int err;
  
  /* get the current swparams */
  err = snd_pcm_sw_params_current(handle, swparams);
  if (err < 0) {
    printf("Unable to determine current swparams for playback: %s\n", snd_strerror(err));
    return err;
  }
  /* allow the transfer when at least period_size samples can be processed */
  /* or disable this mechanism when period event is enabled (aka interrupt like style processing) */
  err = snd_pcm_sw_params_set_avail_min(handle, swparams, dev->period_size);
  if (err < 0) {
    printf("Unable to set avail min for playback: %s\n", snd_strerror(err));
    return err;
  }
  /* enable period events */
  err = snd_pcm_sw_params_set_period_event(handle, swparams, 1);
  if (err < 0) {
    printf("Unable to set period event: %s\n", snd_strerror(err));
    return err;
  }

  /* write the parameters to the playback device */
  err = snd_pcm_sw_params(handle, swparams);
  if (err < 0) {
    printf("Unable to set sw params for playback: %s\n", snd_strerror(err));
    return err;
  }
  return 0;
}

/*
 *   Underrun and suspend recovery
 */
 
int alsa_xrun_recovery(snd_pcm_t *handle, int err)
{
//	printf("stream recovery\n");
	if (err == -EPIPE) {	/* under-run */
		err = snd_pcm_prepare(handle);
		if (err < 0)
			printf("Can't recovery from underrun, prepare failed: %s\n", snd_strerror(err));
		return 0;
	} else if (err == -ESTRPIPE) {
		while ((err = snd_pcm_resume(handle)) == -EAGAIN)
			sleep(1);	/* wait until the suspend flag is released */
		if (err < 0) {
			err = snd_pcm_prepare(handle);
			if (err < 0)
				printf("Can't recovery from suspend, prepare failed: %s\n", snd_strerror(err));
		}
		return 0;
	}
	return err;
}

int alsa_async_direct_loop(alsa_dev_t *dev, void *ptr, void (*callback)(snd_async_handler_t *))
{
  snd_async_handler_t *ahandler;
  snd_pcm_t *phandle = dev->phandle;
  snd_pcm_t *chandle = dev->chandle;
  snd_pcm_uframes_t period_size = dev->period_size;
  pc_data_t *data = (pc_data_t *) ptr;
  PartConvMulti *pc = data->pc;

  const snd_pcm_channel_area_t *my_areas;
  snd_pcm_uframes_t offset, frames;
  snd_pcm_sframes_t avail, commitres;
  int err;

  err = snd_async_add_pcm_handler(&ahandler, phandle, callback, ptr);
  if (err < 0) {
    printf("Unable to register async handler\n");
    exit(-1);
  }
  
  do {  
    avail = snd_pcm_avail_update(phandle);
    if (avail == 0)
      break;
    
//     printf("\nADL: avail_playback == %d\n", avail);

    frames = period_size;
    err = snd_pcm_mmap_begin(phandle, &my_areas, &offset, &frames);
    if (err < 0) {
      printf("MMAP begin avail error: %s\n", snd_strerror(err));
      exit(-1);
    }
    
    if (frames != period_size)
    {
      printf("Error: [ADL] frames != period_size (%d != %d) after snd_pcm_mmap_begin()\n", (int)frames, (int)period_size);
      exit(-1);
    }

    commitres = snd_pcm_mmap_commit(phandle, offset, frames);
    if (commitres < 0 || (snd_pcm_uframes_t)commitres != frames) {
      printf("MMAP commit error: %s\n", snd_strerror(err));
      exit(-1);
    }
    avail = snd_pcm_avail_update(phandle);
  } while (1);

  do
  {
    avail = snd_pcm_avail_update(chandle);
    if (avail == 0)
      break;
    
//     printf("\nADL: avail_capture == %d\n", avail);

    frames = period_size;
    err = snd_pcm_mmap_begin(chandle, &my_areas, &offset, &frames);
    if (err < 0) {
	printf("MMAP begin avail error: %s\n", snd_strerror(err));
	exit(-1);
    }
    
    if (frames != period_size)
    {
      printf("Error: [ADL] frames != period_size (%d != %d) after snd_pcm_mmap_begin()\n", (int)frames, (int)period_size);
      exit(-1);
    }
    
    commitres = snd_pcm_mmap_commit(chandle, offset, frames);
    if (commitres < 0 || (snd_pcm_uframes_t)commitres != frames) {
	printf("MMAP commit error: %s\n", snd_strerror(err));
	exit(-1);
    }
    avail = snd_pcm_avail_update(chandle);
  } while (1);
  
  printf("About to start playback.\n");
  err = snd_pcm_start(phandle);
  if (err < 0) {
    printf("Playback start error: %s\n", snd_strerror(err));
    exit(-1);
  }  
  
  /* because all other work is done in the signal handler,
  suspend the process */
  
  if (pc->lastFrame == 0)
  {
    while(1)
    {
      printf("enter to exit: \n");
      if (getc(stdin) == '\n')
	break;
    }
  }
  else
  {
    printf("running for %d frames\n", pc->lastFrame);
    pc->doneWaiter->waitFor(1);
  }
  
  return 0;
}

int alsa_init(alsa_dev_t *dev, void *ptr, void (*callback)(snd_async_handler_t *))
{
  snd_pcm_t *phandle;
  snd_pcm_t *chandle;
  
  int err;
  snd_pcm_hw_params_t *hwparams_capture, *hwparams_playback;
  snd_pcm_sw_params_t *swparams_capture, *swparams_playback;
    
  struct sched_param param;
  
  err = sched_getparam(0, &param);
  if (err < 0) {
    perror("sched_getparam():");
    exit(-1);
  }
  
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  err = sched_setscheduler(0, SCHED_FIFO, &param);
  if (err < 0) {
    perror("sched_setscheduler():");
   exit(-1);
  }  

  snd_pcm_hw_params_alloca(&hwparams_capture);
  snd_pcm_sw_params_alloca(&swparams_capture);  
  
  snd_pcm_hw_params_alloca(&hwparams_playback);
  snd_pcm_sw_params_alloca(&swparams_playback);   
  
  err = snd_output_stdio_attach(&dev->output, stdout, 0);
  if (err < 0) {
    printf("Output failed: %s\n", snd_strerror(err));
    exit(-1);
  }  
  
  printf("Playback device is %s\n", dev->pdevice);
  printf("Stream parameters are %dHz, %s, %d channels\n", dev->rate, snd_pcm_format_name(dev->format), dev->channels);
  
  if ((err = snd_pcm_open(&phandle, dev->pdevice, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
    printf("Playback open error: %s\n", snd_strerror(err));
    exit(-1);
  }
  
  if ((err = alsa_set_hwparams(dev, phandle, hwparams_playback, SND_PCM_ACCESS_MMAP_INTERLEAVED)) < 0) {
    printf("Setting of hwparams_playback failed: %s\n", snd_strerror(err));
    exit(-1);
  }
  
  if ((err = alsa_set_swparams(dev, phandle, swparams_playback)) < 0) {
    printf("Setting of swparams_playback failed: %s\n", snd_strerror(err));
    exit(-1);
  }
  
  snd_pcm_dump(phandle, dev->output);  
  
  printf("Capture device is %s\n", dev->cdevice);
  
  if ((err = snd_pcm_open(&chandle, dev->cdevice, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    printf("Capture open error: %s\n", snd_strerror(err));
    exit(-1);
  }
  
  if ((err = alsa_set_hwparams(dev, chandle, hwparams_capture, SND_PCM_ACCESS_MMAP_INTERLEAVED)) < 0) {
    printf("Setting of hwparams_capture failed: %s\n", snd_strerror(err));
    exit(-1);
  }
  
  if ((err = alsa_set_swparams(dev, chandle, swparams_capture)) < 0) {
    printf("Setting of swparams_capture failed: %s\n", snd_strerror(err));
    exit(-1);
  }
  
  snd_pcm_dump(chandle, dev->output);    
  
  if ((err = snd_pcm_link(phandle, chandle)) < 0)
  {
    printf("snd_pcm_link() failed: %s\n", snd_strerror(err));
    exit(-1);    
  }
  
  dev->chandle = chandle;
  dev->phandle = phandle;

  err = alsa_async_direct_loop(dev, ptr, callback);
  if (err < 0)
    printf("Transfer failed: %s\n", snd_strerror(err));  
  
  return 0;  
}

void alsa_fini(alsa_dev_t *dev)
{
  printf("Entering alsa_fini()\n"); 
  snd_pcm_close(dev->phandle);
  snd_pcm_close(dev->chandle);
}

#ifdef __cplusplus
}
#endif
