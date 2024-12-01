import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input

class Shape(layers.Layer):
    
    def call(self, x):
        
        return tf.shape(x)

class DataTransformer(layers.Layer):

    def __init__(self, 
            resampled_shape=None, batch_size=None, 
            rand_aff_sca=None, rand_aff_rot=None, rand_aff_tra=None, 
            rand_aff_2d=False, rand_aff_sca_rigid=True, 
            norm_shift=0.0, norm_scale=1.0, norm_lower=None, norm_upper=None,
            rand_shift=(-0.1, +0.1), rand_scale=(0.9, 1.1),
            seed=0, gen=None, **kwargs):
        """
        Object for implementation of key data transformation pipelines

        Key functions:

          1. Resampling (to uniform shape) 
          2. Spatial augmentation (affine-based)
          3. Intensity normalization (and augmentation)

        Overview of arguments based on use-case:

          1. Data augmentation (spatial and/or intensity)

             batch_size      : optional (inferred from input) 
             resampled_shape : optional (inferred from input if not None) 

        ===============================================
        AUGMENTATION
        ===============================================

        Uses: improve generalizability 

        :params

          (tuple) rand_aff_sca       : (lower, upper) range of random zoom scalar 
          (tuple) rand_aff_rot       : (lower, upper) range of random rotation (expressed in radians) 
          (tuple) rand_aff_tra       : (lower, upper) range of random translation (expressed in voxels) 
          (bool)  rand_aff_2d        : apply transform in 2D only, True or False
          (bool)  rand_aff_sca_rigid : apply rigid scalar zoom only, True or False

        NOTE: affine augmentation is automatically adjusted for anisotropy based on size of input

        ===============================================
        NORMALIZATION 
        ===============================================

        Uses: apply intensity normalization

          x = (x.clip(min, max) - shift) / scale

        If augmentation is applied, then:

          shift = scale * rand(rand_shift[0], rand_shift[1]) + shift
          scale = scale * rand(rand_scale[0], rand_scale[1])

        :params

          (float) norm_shift   : shift (raw voxel value) or 'mu'
          (float) norm_scale   : scale (raw voxel value) or 'sd'
          (float) norm_lower   : lower clip
          (float) norm_upper   : upper clip

          (tuple) rand_shift   : (lower, upper) random brightness as % of scale
          (tuple) rand_scale   : (lower, upper) random contrast as % of scale

          (tuple) rand_noise   : (mean , stdev) range of random Gaussian noise
          (tuple) rand_blur    : sigma of random Gaussian blur

        ===============================================
        OTHER GENERAL PARAMETERS 
        ===============================================

        :params

          (int)   batch_size   : number of inputs per batch 
          (int)   seed         : randomization seed 

        """
        super(DataTransformer, self).__init__()
        self.gen = tf.random.Generator.from_seed(seed=seed) if gen is None else gen
        
        # --- Prepare resampler
        self.R_2d = ResamplerND(dims=2)
        self.R_3d = ResamplerND(dims=3)

        # --- Save hyperparameters
        self.resampled_shape = resampled_shape
        self.batch_size = batch_size

        self.rand_aff_sca = rand_aff_sca
        self.rand_aff_rot = rand_aff_rot 
        self.rand_aff_tra = rand_aff_tra 

        self.rand_aff_2d = rand_aff_2d 
        self.rand_aff_sca_rigid = rand_aff_sca_rigid 

        self.rand_shift = rand_shift 
        self.rand_scale = rand_scale 

        self.norm_lower = norm_lower
        self.norm_upper = norm_upper
        self.norm_shift = norm_shift
        self.norm_scale = norm_scale

        # --- Constants
        self.e_2d = np.array([
            0, 0, 0, 0,
            0, 1, 1, 1,
            0, 1, 1, 1,
            0, 1, 1, 1], dtype='float32')

        self.o_2d = np.array([
            1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0], dtype='float32')

        # --- Initializers
        self.init_sampling_parameters()

    def get_config(self):

        config = super().get_config().copy()

        return config

    def get_resampled_shape(self, x=None, **kwargs):

        if self.resampled_shape is not None:
            return self.resampled_shape

        if None in x.shape[1:-1]:
            return Shape()(x)[1:-1]
        else:
            return [s for s in x.shape[1:-1]]

    def init_sampling_parameters(self, **kwargs):
        """
        Method to preparing sampling parameters

        """
        self.n_samples = 1
        self.batch_size_full = self.batch_size 

    def sample_fixed_rate(self, **kwargs):

        pass

    def sample_variable_rate(self, **kwargs):

        # --- Sample random floats 
        r = self.rand(
            shape=(self.n_samples, self.batch_size), 
            minval=0, 
            maxval=self.bins[-1])

        # --- Map random values to mask_val classes
        mask_vals = []
        for k, lo, hi in zip(self.sampling.keys(), self.bins[:-1], self.bins[1:]):
            mask_vals.append(tf.where((r >= lo) & (r < hi), k, 0))
        mask_vals = tf.reduce_sum(mask_vals, axis=0)

        # --- Split mask_vals into interable list
        self.mask_vals = [mask_vals[i] for i in range(self.n_samples)]

    def call(self, x, y=None, l=None, m=None, classes=None, dims=None, affine_fov=None, default_binarize=True, sampling_binarize=False, norms=None, **kwargs):
        """
        Method to apply data transform pipepline

        :params

          (tf.Tensor) x          : input data or dict
          (str)       y          : key(s) in input dict for N-d label(s)
          (str)       l          : key(s) in input dict for 1-d label(s)
          (str)       m          : (1): key (input dict) for sampling mask if >1 labels
          (tf.Tensor)            : (2): tf.Tensor for distinct sampling mask (do not transform)
          (dict)      classes    : mapping for multiclass labels 
                                 : (1): {k0: num_classes, k2...} (num_classes is exclusive of background)
                                 : (2): {k0: {1: v0: 2: v1, ...}, k2...} (provide explicit mapping)
          (tf.Tensor) affine_fov : predefine crop FOV (instead of randomly sampled patches) 

        :shapes

          (tf.Tensor) x          : (N, Z, Y, X, C) ==> data
                                 : (N, Z, Y, X, 1) ==> mask
          (tf.Tensor) dims       : (N, 3)
          (tf.Tensor) affine_fov : (N, 4, 4)

        :return

          x_ : 5D tensor with transformed data or dict

        See __init__() for more details on setting augmentation parameters.

        """
        if y is not None:
            assert type(x) is dict
            assert type(y) in [str, list, tuple]

        if l is not None:
            assert type(x) is dict
            assert type(l) in [str, list, tuple]

            if type(l) is str:
                l = [l]
            l = {k: x.pop(k) for k in l}

        if type(m) is str:
            assert type(x) is dict

        if type(x) is dict:

            # --- Extract norms
            norms = norms or {}
            norms_keys = []
            for k in ['norm_scale', 'norm_shift']:
                for k_, v_ in x.items():
                    if (k in k_) and (k_[:1] == '_'):
                        norms_keys.append(k_)
                        k_ = k_.replace(k, '')[1:-1]
                        if k_ not in norms:
                            norms[k_] = {}
                        norms[k_][k] = v_
            for k in norms_keys:
                x.pop(k)

            for v in x.values():
                assert len(v.shape) == 5
            X = next(iter(x.values()))

        else:
            norms = norms or {}
            assert len(x.shape) == 5
            X = x

        if classes is None:
            classes = {}
        else:
            assert type(x) is dict
            assert type(classes) is dict

            if len(classes) > 0:
                v = next(iter(classes.values()))
                assert type(v) in [dict, int]
                if type(v) is int:
                    classes = {k: {v_: v_ for v_ in range(1, v + 1)} for k, v in classes.items()}
                else:
                    classes = {k: {k_: v[k_] for k_ in sorted(v.keys())} for k, v in classes.items()}

        if dims is not None:
            assert len(dims.shape) == 2
            assert dims.shape[1] == 3

        if affine_fov is not None:
            assert len(affine_fov.shape) == 3
            assert affine_fov.shape[1] == 4
            assert affine_fov.shape[2] == 4

        lbl_keys = []
        if type(x) is dict:
            if type(y) in [list, tuple]:
                lbl_keys = list(y)
            if type(y) is str:
                lbl_keys.append(y)
            if type(m) is str:
                lbl_keys.append(m)

        # ===============================================
        # APPLY INTENSITY-NORMALIZATION ONLY (NO RESAMPLING)
        # ===============================================

        if (self.resampled_shape is None) and \
           (self.rand_aff_sca is None) and \
           (self.rand_aff_rot is None) and \
           (self.rand_aff_tra is None):

            x_ = self.augment_and_normalize(
                x=x,
                l=l,
                norms=norms,
                classes=classes,
                lbl_keys=lbl_keys,
                affine_fov=None,
                default_binarize=default_binarize,
                **kwargs)

            return x_

        # ===============================================
        # APPLY PATCH SAMPLING WITH AFFINE_FOV
        # ===============================================

        if affine_fov is None:

            # ===========================================
            # AFFINE MATRIX SCHEMA
            # ===========================================
            # 
            # ------------------|-----
            # a0[0] a0[1] a0[2] | o[0]
            # a1[0] a1[1] a1[2] | o[1]
            # a2[0] a2[1] a2[2] | o[2]
            # ------------------|-----
            #   1     1     1      1
            # ------------------|-----
            #
            # a0, a1, a2 ==> (batch, 1, 3)
            # o          ==> (batch, 3, 1)
            # ones       ==> (batch, 1, 4)
            # 
            # affine_fov = tf.concat([a0, a1, a2], axis=1)
            # affine_fov = tf.concat([affine_fov, o], axis=2)
            # affine_fov = tf.concat([affine_fov, ones], axis=1)
            # 
            # ===========================================

            # --- STEP 1: Define patch shape ==> rescale affine_fov 
            ZEROES = tf.zeros((self.batch_size_full, 1, 1), dtype=tf.float32) if self.batch_size_full is not None else \
                     tf.zeros((Shape()(X)[0], 1, 1), dtype=tf.float32)
            a0 = np.array([1, 0, 0]).reshape(1, 1, 3) + ZEROES 
            a1 = np.array([0, 1, 0]).reshape(1, 1, 3) + ZEROES 
            a2 = np.array([0, 0, 1]).reshape(1, 1, 3) + ZEROES 

            # --- Assume same as full volume
            if self.resampled_shape is not None:
                scale = tf.cast(Shape()(X)[1:4] / self.resampled_shape, tf.float32)
                a0 = a0 * scale[0]
                a1 = a1 * scale[1]
                a2 = a2 * scale[2]

            # --- No offsets
            o = tf.zeros((1, 3, 1), dtype=tf.float32) + ZEROES 
            h0 = 1.0
            h1 = 0.0

            # --- Assemble affine matrix
            ones = np.array([0, 0, 0, 1]).reshape(1, 1, 4) + ZEROES
            affine_fov = tf.concat([a0, a1, a2], axis=1)
            affine_fov = tf.concat([affine_fov, o - (h1 / h0)], axis=2)
            affine_fov = tf.concat([affine_fov, ones], axis=1)

        # ===============================================
        # APPLY WARP AND PERFORM RESAMPLING 
        # ===============================================

        x_ = self.augment_and_normalize(
            x=x,
            l=l,
            norms=norms,
            classes=classes,
            lbl_keys=lbl_keys,
            affine_fov=affine_fov,
            default_binarize=default_binarize,
            **kwargs)

        return x_

    def augment_and_normalize(self, x, l, norms, classes, lbl_keys, affine_fov=None, default_binarize=True, **kwargs):

        trivial_classes = lambda c : all([k == v for k, v in c.items()])

        # --- Prepare x min values (zero-center prior to resampling)
        if affine_fov is not None:

            if type(x) is dict:

                x_ = {}
                min_values = {}

                for k, v in x.items():

                    # --- Prepare lbl (linear interpolation)
                    if k in lbl_keys:

                        if k in classes:
                            # --- One-hot encoding: multiclass
                            x_[k] = tf.cast(v == np.array(list(classes[k].values())).reshape(1, 1, 1, 1, -1), tf.float32)
                        else: 
                            # --- Default: binary
                            x_[k] = v

                    # --- Prepare dat
                    else:
                        min_values[k] = tf.reduce_min(v)
                        x_[k] = v - min_values[k]
            else:
                min_values = tf.reduce_min(x)
                x_ = x - min_values

            # --- Prepare x shape
            if self.n_samples > 1:
                x_ = self.stack_arr(x=x_) if type(x_) is not dict else \
                    {k: self.stack_arr(x=v) for k, v in x_.items()}

            # --- Perform augmentation
            if affine_fov is not None:
                x_ = self.augment(x=x_, affine_fov=affine_fov, **kwargs)

        else:

            if type(x) is dict:
                x_ = x 
                min_values = {k: 0 for k in x if k not in lbl_keys}
                for k, v in classes.items():
                    if k in x_:
                        if not trivial_classes(v):
                            # --- One-hot encoding: multiclass
                            x_[k] = tf.cast(x[k] == np.array(list(v.values())).reshape(1, 1, 1, 1, -1), tf.float32)

            else:
                x_ = x 
                min_values = 0

        # ===============================================
        # APPLY INTENSITY NORMALIZATION AND AUGMENTATION 
        # ===============================================

        norms = {k: {k_: self.stack_arr(v_) for k_, v_ in v.items()} for k, v in norms.items()}

        if type(x_) is dict:

            # --- Update: dat
            x_.update({k: self.normalize(x_[k], b=min_values[k], norms=norms.get(k), **kwargs) for k in min_values})

            if affine_fov is not None:

                # --- Update: binary labels
                if default_binarize:
                    x_.update({k: tf.cast(x_[k] > 0.5, tf.float32) for k in lbl_keys if k not in classes})

                # --- Update: multiclass labels
                x_.update({k: self.onehot_to_multiclass(arr=x_[k], c=v) for k, v in classes.items() if k in x_})

            else:
                # --- Update: multiclass labels
                for k, v in classes.items():
                    if k in x_:
                        if not trivial_classes(v):
                            x_[k] = self.onehot_to_multiclass(arr=x_[k], c=v)

        else:
            x_ = self.normalize(x_, b=min_values, norms=norms, **kwargs)

        if l is not None:
            l = {k: self.stack_arr(v) for k, v in l.items()}
            x_ = {**x_, **l}

        return x_

    def onehot_to_multiclass(self, arr, c, thresh=0.5, **kwargs):

        if arr.shape[-1] == 1:
            return tf.cast(arr > thresh, tf.float32)

        # --- Perform standard mapping ({1: ..., 2:... , 3: ...})
        msk = tf.cast(tf.math.reduce_sum(arr, axis=-1, keepdims=True) > thresh, tf.float32)
        arg = tf.cast(tf.expand_dims(tf.argmax(arr, axis=-1) + 1, axis=-1), tf.float32)

        # --- Perform alternate mapping ({k0: ..., k1: ..., k2: ...})
        alt = np.array(list(c.keys())).astype('int').reshape(1, 1, 1, 1, -1)
        std = np.arange(1, alt.size + 1).astype('int').reshape(1, 1, 1, 1, -1)
        if not np.all(alt == std):
            arg = tf.cast(arg == std.astype('float32'), tf.int32) * alt
            arg = tf.cast(tf.math.reduce_sum(arg, axis=-1, keepdims=True), tf.float32)

        return arg * msk

    def stack_arr(self, x):

        if self.n_samples in [None, 1]:
            return x

        multiples = (self.n_samples,) + (1,) * (x.shape.rank - 1)

        return tf.tile(x, multiples=multiples)

    def normalize(self, x, b=0, training=False, norms=None, **kwargs):
        """
        Method to apply intensity normalization

          x = (x.clip(min, max) - shift) / scale

        If augmentation is applied, then:

          shift = scale * rand(rand_shift[0], rand_shift[1]) + shift
          scale = scale * rand(rand_scale[0], rand_scale[1])

        :params

          (float) self.norm_shift : shift (raw voxel value) or 'mu'
          (float) self.norm_scale : scale (raw voxel value) or 'sd'
          (float) self.norm_lower : lower clip
          (float) self.norm_upper : upper clip

          (tuple) self.rand_shift : (lower, upper) random brightness as % of scale
          (tuple) self.rand_scale : (lower, upper) random contrast as % of scale

          (float) b               : zero-shift correction (internal)

        """
        norms = norms or {}

        if norms.get('norm_scale') is not None:
            scale = tf.reshape(norms['norm_scale'], [-1, 1, 1, 1, 1]) 
        elif self.norm_scale == 'sd':
            scale = tf.math.reduce_std(x, axis=(1, 2, 3, 4), keepdims=True)
        else:
            scale = float(self.norm_scale)

        if norms.get('norm_shift') is not None:
            shift = tf.reshape(norms['norm_shift'], [-1, 1, 1, 1, 1]) 
        elif self.norm_shift == 'mu':
            shift = tf.math.reduce_mean(x, axis=(1, 2, 3, 4), keepdims=True)
        else:
            shift = float(self.norm_shift)

        if (self.rand_scale is not None) and training:
            rand_scale = self.rand((Shape()(x)[0], 1, 1, 1, 1),
                minval=self.rand_scale[0],
                maxval=self.rand_scale[1], dtype=tf.float32)
        else:
            rand_scale = 1

        if (self.rand_shift is not None) and training:
            rand_shift = self.rand((Shape()(x)[0], 1, 1, 1, 1),
                minval=self.rand_shift[0],
                maxval=self.rand_shift[1], dtype=tf.float32)
        else:
            rand_shift = 0

        _shift = -shift + rand_shift * scale + b
        _scale = scale * rand_scale

        x = (x + _shift) / _scale 

        if self.norm_lower is not None:
            x = tf.clip_by_value(x, self.norm_lower, self.norm_upper)
    
        return x

    def augment(self, x, affine_fov=None, order=1, *args, **kwargs):
        """
        Method to apply 2D/3D data augmentation to input batch x.

        :params

          (tf.Tensor) x          : input data or dict 
          (tf.Tensor) dims       : voxel dimensions 
          (tf.Tensor) affine_fov : affine matrix that centers the transformation to a specific FOV

        :shapes

          (tf.Tensor) x          : (N, Z, Y, X, C)
          (tf.Tensor) dims       : (N, 3)
          (tf.Tensor) affine_fov : (N, 4, 4)

        :return

          x_ : 5D tensor with transformed data or dict

        See __init__() for more details on setting augmentation parameters.

        """
        if type(x) is dict:

            # --- Confirm all volumes in x have the same shape 
            X = next(iter(x.values()))
            assert all([tuple(v.shape[:-1]) == tuple(X.shape[:-1]) for v in x.values()])

        else:
            X = x
            assert len(x.shape) == 5

        # --- Modeify for 2D
        if X.shape[1] == 1:
            self.rand_aff_2d = True

        # --- Create coords 
        s = self.get_resampled_shape(x=X)
        coords = tf.stack(tf.meshgrid(
            tf.range(s[0]), 
            tf.range(s[1]), 
            tf.range(s[2]), 
            indexing='ij'), axis=-1)

        coords = tf.cast(tf.expand_dims(coords, axis=0), X.dtype) + tf.zeros_like(X[:, :1, :1, :1, :1])

        # --- Create affine
        affine_aug = self.rand_affine(shape=Shape()(X), **kwargs)
        if affine_aug.shape[0] == 1:
            affine_aug = affine_aug + tf.zeros_like(X[:, :1, :1, 0, 0])

        # --- Apply affine to coords 
        coords, affine_cmb = self.apply_matmul_coords(coords=coords, affine_aug=affine_aug, affine_fov=affine_fov)

        # --- Prepare resampler
        if X.shape[1] == 1:

            self.R = self.R_2d

            # --- Apply resampler
            if type(x) is not dict:
                x_ = self.R(data=x[:, 0], warp=coords[:, 0, ..., 1:], order=order)
                x_ = tf.expand_dims(x_, axis=1)

            else:
                x_ = resample_dict(
                    data={k: v[:, 0] for k, v in x.items()}, 
                    warp=coords[:, 0, ..., 1:], R=self.R, order=order)
                x_ = {k: tf.expand_dims(v, axis=1) for k, v in x_.items()}

            return x_

        else:

            self.R = self.R_3d

            # --- Apply resampler 
            if type(x) is not dict: 
                return self.R(data=x, warp=coords, order=order)

            else:
                return resample_dict(data=x, warp=coords, R=self.R, order=order)

    def apply_matmul_coords(self, coords, affine_aug, affine_fov=None, transpose_b=True):
        """
        Method to apply coordinate-centered affine transformation to standard (ijk) coords

        """
        if coords.shape[-1] == 3:
            coords = tf.concat((coords, tf.ones_like(coords[..., :1])), axis=-1)

        shape = Shape()(coords)[1:-1]
        ZEROES = tf.zeros_like(affine_aug[:, :1, :1])

        # --- Step 1: seed 
        affine_cmb = tf.eye(4)
        affine_cmb = tf.expand_dims(affine_cmb, axis=0)

        # --- Step 2: convert to provided FOV
        if affine_fov is not None:
            affine_cmb = tf.matmul(affine_fov, affine_cmb)

        # --- Step 3: convert to centered coordinates
        centers = (tf.cast(shape, tf.float32) - 1) / 2

        if affine_fov is not None:
            centers = tf.reshape(tf.concat((centers, [1]), axis=0), [1, 4])
            centers = tf.matmul(centers, affine_fov[:, :3], transpose_b=True)
        else:
            centers = centers + ZEROES

        centers = tf.reshape(centers, (-1, 3, 1))
        centers = tf.concat((tf.zeros((1, 3, 3)) + ZEROES, centers), axis=2)
        centers = tf.concat((centers, tf.zeros((1, 1, 4)) + ZEROES), axis=1)
        centers = centers + (tf.eye(4) + ZEROES)

        affine_cmb = tf.matmul(tf.linalg.inv(centers), affine_cmb)

        # --- Step 4: apply augmentation
        affine_cmb = tf.matmul(affine_aug, affine_cmb)

        # --- Step 5: reverse centered coordinates
        affine_cmb = tf.matmul(centers, affine_cmb)
        
        to_flat = layers.Reshape((-1, 4), dtype=coords.dtype)
        to_norm = lambda x : tf.reshape(x, Shape()(coords[..., :3]))

        coords = to_norm(tf.matmul(to_flat(coords), affine_cmb[:, :3], transpose_b=transpose_b))

        return coords, affine_cmb

    def rand_affine(self, shape, training=False, **kwargs):

        # --- Determine shapes
        batch_shape = shape[0]
        norm_shape = tf.cast(shape[1:4] / tf.reduce_max(shape[1:4]), tf.float32)

        aff = tf.eye(4, batch_shape=[batch_shape])

        if not training:
            return aff

        if self.rand_aff_sca is not None:
            aff = tf.matmul(aff, self.rand_sca(batch_shape=batch_shape))

        if self.rand_aff_tra is not None:
            aff = tf.matmul(aff, self.rand_tra(batch_shape=batch_shape, norm_shape=norm_shape))

        if self.rand_aff_rot is not None:
            aff = tf.matmul(aff, self.rand_rot(batch_shape=batch_shape, norm_shape=norm_shape))

        return aff

    def rand_sca(self, batch_shape, **kwargs):

        e = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype='float32') 
        o = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype='float32')

        a = self.rand(shape=(batch_shape, 16 if not self.rand_aff_sca_rigid else 1), minval=self.rand_aff_sca[0], maxval=self.rand_aff_sca[1]) * e.reshape(1, 16) + o.reshape(1, 16)

        if self.rand_aff_2d:
            a = a * self.e_2d + self.o_2d

        return tf.reshape(a, [batch_shape, 4, 4])

    def rand_tra(self, batch_shape, norm_shape=(1, 1, 1), **kwargs):

        e = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype='float32') 
        o = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype='float32') 

        a = self.rand(shape=(batch_shape, 16), minval=self.rand_aff_tra[0], maxval=self.rand_aff_tra[1]) * e.reshape(1, 16) + o.reshape(1, 16)

        if self.rand_aff_2d:
            a = a * self.e_2d + self.o_2d

        # --- Account for voxel anisotropy
        d = tf.concat((norm_shape, [1]), axis=0)
        d = tf.reshape(d, [1, 4, 1]) 
        d = tf.concat((tf.ones((1, 4, 3), dtype=d.dtype), d), axis=2)

        return tf.reshape(a, [batch_shape, 4, 4]) * tf.cast(d, tf.float32)

    def rand_rot(self, batch_shape, **kwargs):

        if self.rand_aff_2d:
            return self.rand_rot_axi(batch_shape, **kwargs)

        # --- Combine
        axi = self.rand_rot_axi(batch_shape, **kwargs)
        cor = self.rand_rot_cor(batch_shape, **kwargs)
        sag = self.rand_rot_sag(batch_shape, **kwargs)

        return tf.matmul(axi, tf.matmul(cor, sag))

    def rand_rot_sag(self, batch_shape, norm_shape=(1, 1, 1), **kwargs):

        # --- Rotation axis = (0, 1)
        a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        b = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        d = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        o = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype='float32') 

        return self.rand_rot_gen(a=a, b=b, c=c, d=d, o=o, 
            batch_shape=batch_shape, sca=tf.minimum(norm_shape[0], norm_shape[1]))

    def rand_rot_axi(self, batch_shape, norm_shape=(1, 1, 1), **kwargs):

        # --- Rotation axis = (1, 2)
        a = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        b = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype='float32') 
        d = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype='float32') 
        o = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype='float32') 

        return self.rand_rot_gen(a=a, b=b, c=c, d=d, o=o, 
            batch_shape=batch_shape, sca=tf.minimum(norm_shape[1], norm_shape[2]))

    def rand_rot_cor(self, batch_shape, norm_shape=(1, 1, 1), **kwargs):

        # --- Rotation axis = (0, 2)
        a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        b = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype='float32') 
        d = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype='float32') 
        o = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype='float32') 

        return self.rand_rot_gen(a=a, b=b, c=c, d=d, o=o, 
            batch_shape=batch_shape, sca=tf.minimum(norm_shape[0], norm_shape[2]))

    def rand_rot_gen(self, a, b, c, d, o, batch_shape, sca=1.):
        """
        Method to generate random rotation affine matrix in the planes specified by a/b/c/d/o mask

        """
        # --- Generate random rotation for each item in batch
        r = self.rand(shape=(batch_shape, 1), minval=self.rand_aff_rot[0] * sca, maxval=self.rand_aff_rot[1] * sca)

        # --- Calculate affine matrix within masked areas 
        a = tf.math.cos(r) * a.reshape(1, 16)
        b = tf.math.sin(r) * b.reshape(1, 16) 
        c = tf.math.sin(r) * c.reshape(1, 16) * -1.
        d = tf.math.cos(r) * d.reshape(1, 16) 

        return tf.reshape(a + b + c + d + o.reshape(1, 16), [batch_shape, 4, 4])

    def rand(self, shape=(1,), minval=0, maxval=None, dtype=tf.float32, *args, **kwargs):

        return self.gen.uniform(
            shape=shape,
            minval=minval,
            maxval=maxval,
            dtype=dtype)

# =====================================================================
# RESAMPLING
# =====================================================================

class ResamplerND(layers.Layer):

    def __init__(self, dims=3, **kwargs):

        super(ResamplerND, self).__init__()

        self.dims = dims
        self.offsets = tf.constant(self.create_offsets(dims=dims))

    def get_config(self):
        
        config = super().get_config().copy()
        config.update({'dims': self.dims})

        return config

    def call(self, data, warp, order=1, batch_dims=1, clip_warp=False, **kwargs):

        if order == 0:
            return self.resamp_nearest(data, warp, batch_dims, clip_warp=clip_warp)

        if order == 1:
            return self.resamp_linear(data, warp, batch_dims, clip_warp=clip_warp)

    def coords_clip(self, c, data, clip_warp):

        if not clip_warp:
            return c

        return coords_clip(c, lo=0, hi=tuple(np.array(data.shape[1:4]) - 1))

    def resamp_nearest(self, data, warp, batch_dims=1, clip_warp=False):

        c = tf.cast(tf.math.round(warp), tf.int64)
        p = tf.gather_nd(data, self.coords_clip(c, data, clip_warp), batch_dims=batch_dims)

        return p

    def resamp_linear(self, data, warp, batch_dims=1, clip_warp=False):

        # --- Determine gather_nd indices
        f = tf.math.floor(warp)
        c = tf.cast(f, tf.int64)
        for _ in range(self.dims): 
            c = tf.expand_dims(c, -2)

        # --- Gather
        c = self.coords_clip(c + self.offsets, data, clip_warp)
        p = tf.gather_nd(data, c, batch_dims=batch_dims)
        p = tf.cast(p, warp.dtype)

        # --- Linear interpolation
        d = warp - f

        # --- Expand dims
        for _ in range(self.dims):
            d = tf.expand_dims(d, axis=-2)

        if self.dims == 2:

            i = tf.stack((1 - d[..., 0], d[..., 0]), axis=-3)
            j = tf.stack((1 - d[..., 1], d[..., 1]), axis=-2)
            s = i * j
            a = (3, 4)

        else:

            i = tf.stack((1 - d[..., 0], d[..., 0]), axis=-4)
            j = tf.stack((1 - d[..., 1], d[..., 1]), axis=-3)
            k = tf.stack((1 - d[..., 2], d[..., 2]), axis=-2)
            s = i * j * k
            a = (4, 5, 6)

        return tf.math.reduce_sum(p * s, axis=a)

    def create_offsets(self, dims=2):

        if dims == 2:

            b = np.zeros((2, 2, 2), dtype='int')

            b[0, 0] = [0, 0]
            b[1, 0] = [1, 0]
            b[0, 1] = [0, 1]
            b[1, 1] = [1, 1]

            return b.reshape(1, 1, 1, 2, 2, 2)

        else:

            b = np.zeros((2, 2, 2, 3), dtype='int')

            b[0, 0, 0] = [0, 0, 0]
            b[1, 0, 0] = [1, 0, 0]
            b[0, 1, 0] = [0, 1, 0]
            b[0, 0, 1] = [0, 0, 1]
            b[1, 1, 0] = [1, 1, 0]
            b[1, 0, 1] = [1, 0, 1]
            b[0, 1, 1] = [0, 1, 1]
            b[1, 1, 1] = [1, 1, 1]

            return b.reshape(1, 1, 1, 1, 2, 2, 2, 3)

def resample_dict(data, warp, R=None, **kwargs):

    idx_hi = list(np.cumsum([v.shape[-1] for v in data.values()]))
    idx_lo = [0] + idx_hi[:-1]

    c = layers.Concatenate()([v for v in data.values()])

    if R is None:
        R = ResamplerND()

    r = R(data=c, warp=warp, **kwargs)

    return {k: r[..., lo:hi] for k, lo, hi in zip(data.keys(), idx_lo, idx_hi)}

def coords_range(func):

    def wrapper(x, lo=None, hi=None, *args, **kwargs):

        lo = lo or 0
        hi = hi or tuple(np.array(x.shape[1:4]) - 1)

        if type(lo) is not tuple:
            lo = (lo,) * 3

        if type(hi) is not tuple:
            hi = (hi,) * 3

        return func(x=x, lo=lo, hi=hi, *args, **kwargs)

    return wrapper

@coords_range
def coords_clip(x, lo, hi, **kwargs):

    return layers.Concatenate()((
        tf.clip_by_value(x[..., 0:1], lo[0], hi[0]),
        tf.clip_by_value(x[..., 1:2], lo[1], hi[1]),
        tf.clip_by_value(x[..., 2:3], lo[2], hi[2])))

@coords_range
def coords_mask(x, lo, hi, dtype='float32', **kwargs):
    
    valid = lambda x, lo, hi : (x >= lo) & (x <= hi)

    return tf.cast(
        valid(x[..., 0:1], lo[0], hi[0]) &
        valid(x[..., 1:2], lo[1], hi[1]) & 
        valid(x[..., 2:3], lo[2], hi[2]), dtype)
