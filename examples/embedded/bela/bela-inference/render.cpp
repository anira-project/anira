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

std::string gFilename = "ts9_test1_out_FP32.wav";	// Name of the sound file (in project folder)
std::vector<float> gSampleBuffer;				// Buffer that holds the sound file
int gReadPointer = 0;							// Position of the last frame we played

anira::InferenceConfig gInferenceConfig(
	"model.pt",
	{1, 1, 2048},
	{1, 1, 2048},
	21.53f,
	0,
	true,
	0.f,
	false,
	1
);

anira::PrePostProcessor gPrePostProcessor;
anira::InferenceHandler gInferenceHandler(gPrePostProcessor, gInferenceConfig);

float** gAudioData;
int gPrintCounter = 0;

bool setup(BelaContext *context, void *userData)
{
	// Load the sample from storage into a buffer	
	gSampleBuffer = AudioFileUtilities::loadMono(gFilename);
	
	// Check if the load succeeded
	if(gSampleBuffer.size() == 0) {
    	rt_printf("Error loading audio file '%s'\n", gFilename.c_str());
    	return false;
	}

    rt_printf("Loaded the audio file '%s' with %d frames (%.1f seconds)\n", 
    			gFilename.c_str(), gSampleBuffer.size(),
    			gSampleBuffer.size() / context->audioSampleRate);

	anira::HostAudioConfig hostAudioConfig{
		1,
		context->audioFrames,
		context->audioSampleRate
	};

	gInferenceHandler.prepare(hostAudioConfig);
	gInferenceHandler.setInferenceBackend(anira::NONE);
	int latency = gInferenceHandler.getLatency();
	rt_printf("Inference latency: %d samples\n", latency);

	gAudioData = new float*;
	gAudioData[0] = new float[context->audioFrames];

	return true;
}

void render(BelaContext *context, void *userData)
{	
	if (gPrintCounter++ % 1000 == 0) {
		rt_printf("Processing audio\n");
	}
	for(unsigned int n = 0; n < context->audioFrames; n++) {
		gAudioData[0][n] = gSampleBuffer[gReadPointer];

		// Increment and wrap the read pointer
		gReadPointer++;
		if(gReadPointer >= gSampleBuffer.size()) {
			gReadPointer = 0;
		}
	}
	gInferenceHandler.process(gAudioData, context->audioFrames);

	if(gPrintCounter % 1000 == 0) {
		rt_printf("Finished processing audio\n");
	}

	if(gPrintCounter++ > 10000) {
		gPrintCounter = 0;
	}

	for(unsigned int channel = 0; channel < context->audioInChannels; channel++) {
		for(unsigned int n = 0; n < context->audioFrames; n++) {
			// Write the sample to every audio output channel
			audioWrite(context, n, channel, gAudioData[0][n]);
		}
	}
}

void cleanup(BelaContext *context, void *userData)
{
	delete[] gAudioData[0];
	delete[] gAudioData;
}
