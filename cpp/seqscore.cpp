/*
Install conda and download faiss source code, then
On Linux:
g++ -O3 -I ../../faiss -shared -fPIC -fopenmp seqscore.cpp ~/miniconda3/lib/libfaiss_avx2.so -o seqscore

On Windows:
cl /O2 /I ../../faiss /EHsc /LD /openmp seqscore.cpp %HomePath%\Miniconda3\Library\lib\faiss_avx2.lib /Feseqscore
*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <faiss/Index.h>

#ifndef _WIN32
#define __declspec(x)
#endif

int idx_to_song_id(const int64_t *song_pos, int n_songs, int64_t idx) {
    return std::upper_bound(song_pos, song_pos + n_songs, idx) - song_pos - 1;
}

extern "C" __declspec(dllexport)
int seq_score(
        void *index,
        const int64_t *song_pos,
        int n_songs,
        const float *query,
        int query_len,
        const int64_t *labels,
        int top_k,
        float *song_scores,
        int shift,
        int frame_shift_mul)
{
    const faiss::Index *idx = (faiss::Index *) index;
    const int d = idx->d;
    std::vector<std::pair<int,int> > candidates;
    
    for (int t = 0; t < query_len; t++) {
        for (int i = 0; i < top_k; i++) {
            if (labels[t*top_k+i] < 0) continue;

            int song_id = idx_to_song_id(song_pos, n_songs, labels[t*top_k+i]);
            candidates.emplace_back(song_id, int(labels[t*top_k+i] - song_pos[song_id] - t));
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.resize(std::unique(candidates.begin(), candidates.end()) - candidates.begin());
    
    float best = -INFINITY;
    int best_song = -1;
    std::vector<float> tmp_score(candidates.size());
    std::vector<float> tmp_t(candidates.size());
    
    #pragma omp parallel
    {
        std::vector<float> vec(d);
        float my_best = -INFINITY;
        int my_best_song = -1;
        #pragma omp for
        for (int i = 0; i < candidates.size(); i++) {
            int song_id = candidates[i].first;
            if (song_id >= n_songs || song_id < 0) continue;
            int song_len = song_pos[song_id+1] - song_pos[song_id];
            int64_t song_start = song_pos[song_id];
            int t = candidates[i].second;

            float sco = 0;
            for (int j = 0; j < query_len; j++) {
                if (t+j < 0 || t+j >= song_len) continue;
                idx->reconstruct(song_start + t+j, vec.data());
                for (int k = 0; k < d; k++) {
                    sco += vec[k] * query[j*d + k];
                }
            }
            sco /= query_len;
            tmp_score[i] = sco;
            tmp_t[i] = t * frame_shift_mul - shift;
            if (sco > my_best) {
                my_best = sco;
                my_best_song = song_id;
            }
        }
        #pragma omp critical
        if (my_best > best || my_best == best && my_best_song < best_song) {
            best = my_best;
            best_song = my_best_song;
        }
    }
    for (int i = 0; i < candidates.size(); i++) {
        int song_id = candidates[i].first;
        if (song_id >= n_songs || song_id < 0) continue;
        if (tmp_score[i] > song_scores[song_id*2]) {
            song_scores[song_id*2] = tmp_score[i];
            song_scores[song_id*2+1] = tmp_t[i];
        }
    }
    return best_song;
}
