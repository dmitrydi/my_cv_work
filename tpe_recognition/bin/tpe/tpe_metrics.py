import numpy as np

def get_scores(data_y, protocol):
    data_y_norm = data_y / np.linalg.norm(data_y, axis=1)[:, np.newaxis]
    scores = data_y_norm @ data_y_norm.T

    return scores[protocol==True], scores[np.logical_not(protocol)]


def calc_metrics(targets_scores, imposter_scores):
    min_score = np.minimum(np.min(targets_scores), np.min(imposter_scores))
    max_score = np.maximum(np.max(targets_scores), np.max(imposter_scores))
    print('min score: {}, max score: {}'.format(min_score, max_score))

    n_tars = len(targets_scores)
    n_imps = len(imposter_scores)

    N = 100

    fars = np.zeros((N,))
    frrs = np.zeros((N,))
    dists = np.zeros((N,))

    min_gap = float('inf')
    eer = 0

    for i, dist in enumerate(np.linspace(min_score, max_score, N)):
        far = len(np.where(imposter_scores > dist)[0]) / n_imps
        frr = len(np.where(targets_scores < dist)[0]) / n_tars
        fars[i] = far
        frrs[i] = frr
        dists[i] = dist

        gap = np.abs(far - frr)

        if gap < min_gap:
            min_gap = gap
            eer = (far + frr) / 2
            deer = dist

    return eer, fars, frrs, dists, deer