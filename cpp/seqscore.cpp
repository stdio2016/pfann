/*
Install conda and download faiss source code, then
On Linux:
g++ -O3 -I ../../faiss -shared -fPIC -fopenmp seqscore.cpp ~/miniconda3/lib/libfaiss_avx2.so -o seqscore

On Windows:
cl /O2 /I ../../faiss /EHsc /LD /openmp seqscore.cpp %HomePath%\Miniconda3\Library\lib\faiss_avx2.lib /Feseqscore

On Mac miniforge: please use conda compilers to compile faiss
clang++ -O3 -I ../../faiss -shared -fPIC -Xclang -fopenmp  seqscore.cpp ../../faiss/build/faiss/python/_swigfaiss.so -L $CONDA_PREFIX/lib -l omp -std=c++11 -o seqscore
*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <tuple>
#include <faiss/Index.h>

#ifndef _WIN32
#define __declspec(x)
#endif

int idx_to_song_id(const int64_t *song_pos, int n_songs, int64_t idx) {
    return std::upper_bound(song_pos, song_pos + n_songs, idx) - song_pos - 1;
}

extern "C" __declspec(dllexport)
long long version() {
    return 20220625002LL;
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
        int frame_shift_mul,
        float score_alpha)
{
    const faiss::Index *idx = (faiss::Index *) index;
    const int d = idx->d;
    std::vector<std::tuple<int,int,int> > candidates;
    
    for (int t = 0; t < query_len; t++) {
        int tim = t / frame_shift_mul;
        int shift = t % frame_shift_mul;
        for (int i = 0; i < top_k; i++) {
            if (labels[t*top_k+i] < 0) continue;

            int song_id = idx_to_song_id(song_pos, n_songs, labels[t*top_k+i]);
            candidates.emplace_back(song_id, int(labels[t*top_k+i] - song_pos[song_id] - tim), shift);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.resize(std::unique(candidates.begin(), candidates.end()) - candidates.begin());
    
    float best = -INFINITY;
    int best_song = -1;
    std::vector<float> tmp_score(candidates.size());
    std::vector<float> tmp_t(candidates.size());

    int64_t mod = 1;
    while (mod < query_len) {
        mod *= 2;
    }
    
    #pragma omp parallel
    {
        std::vector<int64_t> cache(mod, -1);
        std::vector<float> vec(d * mod);
        float my_best = -INFINITY;
        int my_best_song = -1;
        #pragma omp for
        for (int i = 0; i < candidates.size(); i++) {
            int song_id = std::get<0>(candidates[i]);
            if (song_id >= n_songs || song_id < 0) continue;
            int song_len = song_pos[song_id+1] - song_pos[song_id];
            int64_t song_start = song_pos[song_id];
            int t = std::get<1>(candidates[i]);
            int shift = std::get<2>(candidates[i]);

            float sco = 0;
            int my_query_len = (query_len - shift + frame_shift_mul - 1) / frame_shift_mul;
            for (int j = 0; j < my_query_len; j++) {
                int query_idx = j * frame_shift_mul + shift;
                if (t+j < 0 || t+j >= song_len) continue;
                int64_t song_at = song_start + t+j;
                int64_t song_at_hash = song_at & (mod - 1);
                float *my_vec = &vec[song_at_hash * d];
                if (cache[song_at_hash] != song_at) {
                    idx->reconstruct(song_at, my_vec);
                    cache[song_at_hash] = song_at;
                }
                float innerprod = 0;
                for (int k = 0; k < d; k++) {
                    innerprod += my_vec[k] * query[query_idx*d + k];
                }
                // reference paper: Query adaptive similarity for large scale object retrieval
                // by D. Qin, C. Wengert, and L. V. Gool.
                float l2norm = 1.0f - 1.0f * innerprod;
                if (score_alpha == 0.0f) {
                    sco += innerprod;
                } else if (score_alpha > 0.0f) {
                    sco += expf(-score_alpha * l2norm * l2norm);
                }
            }
            sco /= std::max(my_query_len, 1);
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
        int song_id = std::get<0>(candidates[i]);
        if (song_id >= n_songs || song_id < 0) continue;
        if (tmp_score[i] > song_scores[song_id*2]) {
            song_scores[song_id*2] = tmp_score[i];
            song_scores[song_id*2+1] = tmp_t[i];
        }
    }
    //printf("hit %lld miss %lld\n", hit, miss);
    return best_song;
}
