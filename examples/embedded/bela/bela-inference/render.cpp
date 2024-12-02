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

std::string g_filename = "ts9_test1_out_FP32.wav";	// Name of the sound file (in project folder)
std::vector<float> g_sample_buffer;				// Buffer that holds the sound file
int g_read_pointer = 0;							// Position of the last frame we played

anira::InferenceConfig g_inference_config(
	"model.pt",
	{{1, 1, 2048}},
	{{1, 1, 2048}},
	21.53f,
	0,
	true,
	0.f,
	false,
	1
);

anira::PrePostProcessor g_pp_processor;
anira::InferenceHandler g_inference_handler(g_pp_processor, g_inference_config);

float** audio_data;
int gPrintCounter = 0;

bool setup(BelaContext *context, void *userData)
{
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

	anira::HostAudioConfig host_config{
		1,
		context->audioFrames,
		context->audioSampleRate
	};

	g_inference_handler.prepare(host_config);
	g_inference_handler.set_inference_backend(anira::CUSTOM);
	int latency = g_inference_handler.get_latency();
	rt_printf("Inference latency: %d samples\n", latency);

	audio_data = new float*;
	audio_data[0] = new float[context->audioFrames];

	return true;
}

void render(BelaContext *context, void *userData)
{	
	if (gPrintCounter++ % 1000 == 0) {
		rt_printf("Processing audio\n");
	}
	for(unsigned int n = 0; n < context->audioFrames; n++) {
		audio_data[0][n] = g_sample_buffer[g_read_pointer];

		// Increment and wrap the read pointer
		g_read_pointer++;
		if(g_read_pointer >= g_sample_buffer.size()) {
			g_read_pointer = 0;
		}
	}
	g_inference_handler.process(audio_data, context->audioFrames);

	if(gPrintCounter % 1000 == 0) {
		rt_printf("Finished processing audio\n");
	}

	if(gPrintCounter++ > 10000) {
		gPrintCounter = 0;
	}

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
