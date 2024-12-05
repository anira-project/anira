/*
 ____  _____ _        _    
| __ )| ____| |      / \   
|  _ \|  _| | |     / _ \  
| |_) | |___| |___ / ___ \ 
|____/|_____|_____/_/   \_\

http://bela.io

Anira x Bela Example
*/

#include <Bela.h>
#include <libraries/AudioFile/AudioFile.h>
#include <anira/anira.h>
#include "../SimpleGainConfig.h"

std::string g_filename = "ts9_test1_out_FP32.wav";	// Name of the sound file (in project folder)
std::vector<float> g_sample_buffer;				// Buffer that holds the sound file
int g_read_pointer = 0;							// Position of the last frame we played

anira::InferenceConfig g_inference_config = gain_config;
anira::PrePostProcessor g_pp_processor(g_inference_config);
anira::InferenceHandler g_inference_handler(g_pp_processor, g_inference_config);

float** audio_data;

bool setup(BelaContext *context, void *userData)
{
	rt_printf("Anira x Bela Example\n");
	rt_printf("Current buffer size: %d\n", context->audioFrames);
	rt_printf("Current sample rate: %d\n", context->audioSampleRate);

	// Allocate memory for the audio data
	audio_data = new float*[1];
	audio_data[0] = new float[context->audioFrames];

	// Load the sample from storage into a buffer	
	g_sample_buffer = AudioFileUtilities::loadMono(g_filename);
	
	// Check if the load succeeded
	if(g_sample_buffer.size() == 0) {
    	rt_printf("Error loading audio file '%s'\n", g_filename.c_str());
    	return false;
	}

    rt_printf("Loaded the audio file '%s' with %d frames (%.1f seconds)\n", 
    			g_filename.c_str(), g_sample_buffer.size(),
    			g_sample_buffer.size() / context->audioSampleRate);

	// Prepare the inference handler and set the inference backend
	g_inference_handler.prepare({(size_t)context->audioFrames, context->audioSampleRate});
	g_inference_handler.set_inference_backend(anira::ONNX);

	// Get the latency introduced by the inference handler (in samples)
	int latency = g_inference_handler.get_latency();

	// This model takes a gain value as additional input parameter
	g_pp_processor.set_input(0.5, 1, 0);

	// Some printouts to check
	if(g_inference_handler.get_inference_backend() == anira::LIBTORCH) {
		rt_printf("Using LibTorch backend\n");
	} else if(g_inference_handler.get_inference_backend() == anira::ONNX) {
		rt_printf("Using ONNXRuntime backend\n");
	} else if(g_inference_handler.get_inference_backend() == anira::TFLITE) {
		rt_printf("Using TFLite backend\n");
	} else if(g_inference_handler.get_inference_backend() == anira::CUSTOM) {
		rt_printf("Using custom backend\n");
	} else {
		rt_printf("Backend not selected\n");
	}
	rt_printf("Anira introduces a latency of %d samples\n", latency);

	return true;
}

void render(BelaContext *context, void *userData)
{	
	// Read the audio file and write it to the audio data buffer
	for(unsigned int n = 0; n < context->audioFrames; n++) {
		audio_data[0][n] = g_sample_buffer[g_read_pointer];
		// Increment and wrap the read pointer
		g_read_pointer++;
		if(g_read_pointer >= g_sample_buffer.size()) {
			g_read_pointer = 0;
		}
	}

	// Process the audio data through the model
	g_inference_handler.process(audio_data, context->audioFrames);

	// Write the processed audio data to the audio output
	for(unsigned int channel = 0; channel < context->audioInChannels; channel++) {
		for(unsigned int n = 0; n < context->audioFrames; n++) {
			// Write the sample to every audio output channel
			audioWrite(context, n, channel, audio_data[0][n]);
		}
	}
}

void cleanup(BelaContext *context, void *userData)
{
	delete[] audio_data[0];
	delete[] audio_data;
}
