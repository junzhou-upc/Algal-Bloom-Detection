# Copyright (C) 2008-2016, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)


import numpy as np
from mahotas.features import _texture
from mahotas.internal import _verify_is_integer_type

__all__ = [
    'haralick',
    'haralick_labels',
    'cooccurence',
    ]

def _entropy(p):
    p = p.ravel()
    p1 = p.copy()
    p1 += (p==0)
    return -np.dot(np.log2(p1), p)


def haralick(f,
            ignore_zeros=False,
            preserve_haralick_bug=False,
            compute_14th_feature=False,
            return_mean=False,
            return_mean_ptp=False,
            use_x_minus_y_variance=False,
            distance=1
            ):

    _verify_is_integer_type(f, 'mahotas.haralick')

    if len(f.shape) == 2:
        nr_dirs = len(_2d_deltas)
    elif len(f.shape) == 3:
        nr_dirs = len(_3d_deltas)
    else:
        raise ValueError('mahotas.texture.haralick: Can only handle 2D and 3D images.')
    fm1 = f.max() + 1
    cmat = np.empty((fm1, fm1), np.int32)
    def all_cmatrices():
        for dir in range(nr_dirs):
            cooccurence(f, dir, cmat, symmetric=True, distance=distance)
            yield cmat
    return haralick_features(all_cmatrices(),
                        ignore_zeros=ignore_zeros,
                        preserve_haralick_bug=preserve_haralick_bug,
                        compute_14th_feature=compute_14th_feature,
                        return_mean=return_mean,
                        return_mean_ptp=return_mean_ptp,
                        use_x_minus_y_variance=use_x_minus_y_variance,
                        )

def haralick_features(cmats,
                    ignore_zeros=False,
                    preserve_haralick_bug=False,
                    compute_14th_feature=False,
                    return_mean=False,
                    return_mean_ptp=False,
                    use_x_minus_y_variance=False,
                    ):

    if return_mean and return_mean_ptp:
        raise ValueError("mahotas.haralick_features: Cannot set both `return_mean` and `return_mean_ptp`")
    features = []
    for cmat in cmats:
        feats = np.zeros(13 + bool(compute_14th_feature), np.double)
        if ignore_zeros:
            cmat[0] = 0
            cmat[:,0] = 0
        T = cmat.sum()
        if not T:
            raise ValueError('mahotas.haralick_features: the input is empty. Cannot compute features!\n' +
                                'This can happen if you are using `ignore_zeros`' )
        if not len(features):
            maxv = len(cmat)
            k = np.arange(maxv)
            k2 = k**2
            tk = np.arange(2*maxv)
            tk2 = tk**2
            i,j = np.mgrid[:maxv,:maxv]
            ij = i*j
            i_j2_p1 = (i - j)**2
            i_j2_p1 += 1
            i_j2_p1 = 1. / i_j2_p1
            i_j2_p1 = i_j2_p1.ravel()
            px_plus_y = np.empty(2*maxv, np.double)
            px_minus_y = np.empty(maxv, np.double)
        elif maxv != len(cmat):
            raise ValueError('mahotas.haralick_features: All cmatrices must be of the same size')

        p = cmat / float(T)
        pravel = p.ravel()
        px = p.sum(0)
        py = p.sum(1)

        ux = np.dot(px, k)
        uy = np.dot(py, k)
        vx = np.dot(px, k2) - ux**2
        vy = np.dot(py, k2) - uy**2

        sx = np.sqrt(vx)
        sy = np.sqrt(vy)
        px_plus_y.fill(0)
        px_minus_y.fill(0)
        _texture.compute_plus_minus(p, px_plus_y, px_minus_y)

        feats[0] = np.dot(pravel, pravel)
        feats[1] = np.dot(k2, px_minus_y)

        if sx == 0. or sy == 0.:
            feats[2] = 1.
        else:
            feats[2] = (1. / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy)

        feats[3] = vx
        feats[4] = np.dot(i_j2_p1, pravel)
        feats[5] = np.dot(tk, px_plus_y)

        feats[7] = _entropy(px_plus_y)

        # There is some confusion w.r.t. feats[6].
        #
        # Haralick's paper uses feats[7] in its computation, but it is
        # clear that feats[5] should be used (i.e., it computes a
        # variance).
        #
        if preserve_haralick_bug:
            feats[6] = ((tk-feats[7])**2*px_plus_y).sum()
        else:
            feats[6] = np.dot(tk2, px_plus_y) - feats[5]**2

        feats[ 8] = _entropy(pravel)
        feats[ 9] = 0
        feats[10] = 0
        feats[11] = 0
        feats[12] = 0

        features.append(feats)

    features = np.array(features)
    if return_mean:
        return features.mean(axis=0)
    if return_mean_ptp:
        mean = features.mean(axis=0)
        ptp = features.ptp(axis=0)
        return np.concatenate((mean,ptp))

    return features


haralick_labels = ["Angular Second Moment",
                   "Contrast",
                   "Correlation",
                   "Sum of Squares: Variance",
                   "Inverse Difference Moment",
                   "Sum Average",
                   "Sum Variance",
                   "Sum Entropy",
                   "Entropy",
                   "Difference Variance",
                   "Difference Entropy",
                   "Information Measure of Correlation 1",
                   "Information Measure of Correlation 2",
                   "Maximal Correlation Coefficient"]

_2d_deltas= [
    (0,1)]

_3d_deltas = [
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (1,-1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (1,-1, 1),
    (1, 0,-1),
    (0, 1,-1),
    (1, 1,-1),
    (1,-1,-1) ]

def cooccurence(f, direction, output=None, symmetric=True, distance=1):

    _verify_is_integer_type(f, 'mahotas.cooccurence')
    if len(f.shape) == 2 and not (0 <= direction < 4):
        raise ValueError('mahotas.texture.cooccurence: `direction` {0} is not in range(4).'.format(direction))
    elif len(f.shape) == 3 and not (0 <= direction < 13):
        raise ValueError('mahotas.texture.cooccurence: `direction` {0} is not in range(13).'.format(direction))
    elif len(f.shape) not in (2,3):
        raise ValueError('mahotas.texture.cooccurence: cannot handle images of %s dimensions.' % len(f.shape))

    if output is None:
        mf = f.max()
        output = np.zeros((mf+1, mf+1), np.int32)
    else:
        assert np.min(output.shape) >= f.max(), 'mahotas.texture.cooccurence: output is not large enough'
        assert output.dtype == np.int32, 'mahotas.texture.cooccurence: output is not of type np.int32'
        output.fill(0)

    if len(f.shape) == 2:
        mask_size = 2 * distance + 1
        Bc = np.zeros((mask_size, mask_size), f.dtype)
        y, x = tuple(distance * i for i in _2d_deltas[direction])
        Bc[y + distance, x + distance] = 1
    else:
        mask_size = 2 * distance + 1
        Bc = np.zeros((mask_size, mask_size, mask_size), f.dtype)
        y, x, z = tuple(distance * i for i in _3d_deltas[direction])
        Bc[y + distance, x + distance, z + distance] = 1
    _texture.cooccurence(f, output, Bc, symmetric)
    return output

