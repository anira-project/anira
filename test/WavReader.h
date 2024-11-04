#pragma once
#ifndef ANIRA_WAVREADER_H
#define ANIRA_WAVREADER_H
// Adapted from https://stackoverflow.com/a/75704890
#include <iostream>
#include <cstdint>
#include <fstream>
#include <vector>
#include <cstring>

using namespace std;

struct RIFFHeader{
    char chunk_id[4];
    uint32_t chunk_size;
    char format[4];
};

struct ChunkInfo{
    char chunk_id[4];
    uint32_t chunk_size;
};

struct FmtChunk{
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};


struct DataChunk
// We assume 16-bit monochannel samples
{  
    float* data;
    int nb_of_samples;
    DataChunk(int s): nb_of_samples{s}, data{new float[s]} {}
    ~DataChunk(){delete[] data;}
};

inline int read_wav(string path, std::vector<float>& data){
    constexpr char riff_id[4] = {'R','I','F','F'};
    constexpr char format[4] = {'W','A','V','E'};
    constexpr char fmt_id[4] = {'f','m','t',' '};
    constexpr char data_id[4] = {'d','a','t','a'};

    ifstream ifs{path, ios_base::binary};
    if (!ifs){
        cerr << "Cannot open file " << path << endl;
        return -1;
    }

    // first read RIFF header
    RIFFHeader h;
    ifs.read((char*)(&h), sizeof(h));
    if (!ifs || memcmp(h.chunk_id, riff_id, 4) || memcmp(h.format, format, 4)){
        cerr << "Bad formatting" << endl;
        return -1;
    }

    // read chunk infos iteratively
    ChunkInfo ch;
    bool fmt_read = false;
    bool data_read = false;
    while(ifs.read((char*)(&ch), sizeof(ch))){

        // if fmt chunk?
        if (memcmp(ch.chunk_id, fmt_id, 4) == 0){
            FmtChunk fmt;
            ifs.read((char*)(&fmt), ch.chunk_size);
            fmt_read = true;
        }
        // is data chunk?
        else if(memcmp(ch.chunk_id, data_id, 4) == 0){
            DataChunk dat_chunk(ch.chunk_size/sizeof(float));
            ifs.read((char*)dat_chunk.data, ch.chunk_size);
            std::cout << "dat_chunk size: " << dat_chunk.nb_of_samples << std::endl;
            // put data in vector
            data.assign(dat_chunk.data, dat_chunk.data + dat_chunk.nb_of_samples);

            data_read = true;
        }
        // otherwise skip the chunk
        else{
            ifs.seekg(ch.chunk_size, ios_base::cur);
        }
    }
    if (!data_read || !fmt_read){
        cout << "Problem when reading data" << endl;
        return -1;
    }
    return 0;
}

#endif // ANIRA_WAVREADER_H