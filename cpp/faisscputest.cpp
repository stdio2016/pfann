/*
Install conda and download faiss source code, then
On Linux:
g++ -O3 -I ../../faiss faisscputest.cpp ~/miniconda3/lib/libfaiss_avx2.so -o faisscputest
or nvcc for GPU acceleration

On Windows:
cl /O2 /I ../../faiss /EHsc faisscputest.cpp %HomePath%\Miniconda3\Library\lib\faiss_avx2.lib /Fefaisscputest
*/
#ifdef __NVCC__
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

#include <faiss/index_io.h>
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissException.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

using faiss::Index;

int idx_to_song_id(const int64_t *song_pos, int n_songs, int64_t idx) {
    return std::upper_bound(song_pos, song_pos + n_songs, idx) - song_pos - 1;
}

void my_search(const Index *idx, const int64_t *song_pos, int n_songs, int len, const float *query, const Index *idx2) {
    if (len > 100) len = 100;

    int k = 100;
    int d = idx->d;

    std::vector<int64_t> labels(len * k);
    std::vector<float> distances(len * k);
    std::vector<std::pair<int,int> > candidates;
    std::vector<float> vec(d);

    idx->search(len, query, k, distances.data(), labels.data());

    for (int t = 0; t < len; t++) {
        for (int i = 0; i < k; i++) {
            if (labels[t*k+i] < 0) continue;

            int song_id = idx_to_song_id(song_pos, n_songs, labels[t*k+i]);
            candidates.emplace_back(song_id, int(labels[t*k+i] - song_pos[song_id] - t));
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.resize(std::unique(candidates.begin(), candidates.end()) - candidates.begin());

    float best = -len - 1;
    int best_song = -1;
    for (auto c : candidates) {
        int song_id = c.first;
        int song_len = song_pos[song_id+1] - song_pos[song_id];
        int64_t song_start = song_pos[song_id];
        int t = c.second;

        float sco = 0;
        for (int i = 0; i < len; i++) {
            if (t+i < 0 || t+i >= song_len) continue;
            idx2->reconstruct(song_start + t+i, vec.data());
            for (int j = 0; j < d; j++) {
                sco += vec[j] * query[i*d + j];
            }
        }
        if (sco > best) {
            best = sco;
            best_song = song_id;
        }
    }
    fwrite(&best_song, 4, 1, stdout);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc < 2) return 1;

    faiss::Index *index = NULL;
    std::string filename;
    filename = std::string(argv[1]) + "/landmarkValue";
    
#ifdef __NVCC__
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuClonerOptions opt;
    opt.useFloat16 = true;
#endif

    faiss::Index *index2 = NULL;
    try {
        index2 = index = faiss::read_index(filename.c_str());
        #ifdef __NVCC__
        index2 = faiss::gpu::index_cpu_to_gpu(&res, 0, index, &opt);
        #endif
    }
    catch (faiss::FaissException x) {
         puts(x.what());
         return 1;
    }

    filename = std::string(argv[1]) + "/landmarkKey";
    std::vector<int64_t> song_pos(1);
    FILE *fin = fopen(filename.c_str(), "rb");
    if (!fin) {
         printf("database corrupt!\n");
         return 1;
    }
    int32_t tmp;
    while (fread(&tmp, 4, 1, fin) == 1) {
        song_pos.push_back(song_pos.back() + tmp);
    }
    int n_songs = song_pos.size() - 1;
    fclose(fin);

    //printf("I read %lld data!\n", index->ntotal);
    if (faiss::IndexIVF *ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
        ivf->make_direct_map();
        //ivf->nprobe = ivf->invlists->nlist;
        ivf->nprobe = 50;
    }
    #ifdef __NVCC__
    if (faiss::gpu::GpuIndexIVF *ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(index2)) {
        ivf->setNumProbes(50);
    }
    #endif
    int d = index->d;
    uint32_t len;
    while (fread(&len, 4, 1, stdin) == 1) {
        if (len % d != 0) return 1;
        std::vector<float> query(len);
        uint32_t actual = fread(query.data(), 4, len, stdin);
        //if (len > d * 100) len = d * 100;
        //printf("cpu: ");
        //my_search(index, song_pos.data(), n_songs, len/d, query.data(), index);
        //printf("gpu: ");
        my_search(index2, song_pos.data(), n_songs, len/d, query.data(), index);
    }
}
