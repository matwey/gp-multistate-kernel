import unittest
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from multistate_kernel import util


class MutliStateDataUnitTest(unittest.TestCase):
    def setUp(self):
        self.n = 10

        self.key1 = 'one'
        self.x1 = np.arange(self.n)
        self.y1 = np.zeros(self.n)
        self.err1 = self.y1 / self.n

        self.key2 = 'two'
        self.x2 = np.arange(self.n)
        self.y2 = np.ones(self.n)
        self.err2 = self.y2 / self.n

    def check_msd_from_dict(self, msd):
        assert_equal(msd.odict[self.key1].x, self.x1)
        assert_equal(msd.odict[self.key1].y, self.y1)
        assert_equal(msd.odict[self.key1].err, self.err1)
        assert_equal(msd.odict[self.key2].x, self.x2)
        assert_equal(msd.odict[self.key2].y, self.y2)
        assert_equal(msd.odict[self.key2].err, self.err2)

        assert_equal(msd.arrays.x[:, 0], np.r_[np.zeros(self.n), np.ones(self.n)])
        assert_equal(msd.arrays.x[:, 1], np.r_[self.x1, self.x2])
        assert_allclose(msd.arrays.y * msd.arrays.norm, np.r_[self.y1, self.y2])
        assert_allclose(msd.arrays.err * msd.arrays.norm, np.r_[self.err1, self.err2])

    def get_msd(self):
        items = [(self.key1, [self.x1, self.y1, self.err1]),
                 (self.key2, [self.x2, self.y2, self.err2])]
        msd = util.data_from_items(items)
        return msd

    def test_from_items(self):
        items = [(self.key1, [self.x1, self.y1, self.err1]),
                 (self.key2, [self.x2, self.y2, self.err2])]
        msd = util.data_from_items(items)
        self.check_msd_from_dict(msd)

    def test_from_dict_of_state_data(self):
        state_data1 = util.StateData(x=self.x1, y=self.y1, err=self.err1)
        state_data2 = util.StateData(self.x2, self.y2, self.err2)
        d = OrderedDict([(self.key1, state_data1), (self.key2, state_data2)])
        msd = util.data_from_state_data(d)
        self.check_msd_from_dict(msd)

    def test_from_dict_of_rec_arrays(self):
        rec1 = np.rec.fromarrays([self.x1, self.y1, self.err1], names='x,y,err')
        rec2 = np.rec.fromarrays([self.x2, self.y2, self.err2], names='x,y,err')
        d = OrderedDict([(self.key1, rec1), (self.key2, rec2)])
        msd = util.data_from_state_data(d)
        self.check_msd_from_dict(msd)

    def test_from_arrays(self):
        x_0 = np.r_[np.zeros(self.n), np.ones(self.n)]
        x_1 = np.r_[self.x1, self.x2]
        x = np.stack((x_0, x_1), axis=1)
        y = np.r_[self.y1, self.y2]
        err = np.r_[self.err1, self.err2]
        norm = 2
        
        msd = util.data_from_arrays(x, y, err, norm=norm, keys=[self.key1, self.key2])

        assert_equal(msd.arrays.x[:,0], x_0)
        assert_equal(msd.arrays.x[:,1], x_1)
        assert_equal(msd.arrays.y, y)
        assert_equal(msd.arrays.err, err)
        assert_equal(msd.arrays.norm, norm)
        assert_equal(msd.norm, norm)

        assert_equal(msd.odict[self.key1].x, self.x1)
        assert_allclose(msd.odict[self.key1].y / msd.norm, self.y1)
        assert_allclose(msd.odict[self.key1].err / msd.norm, self.err1)
        assert_equal(msd.odict[self.key2].x, self.x2)
        assert_allclose(msd.odict[self.key2].y / msd.norm, self.y2)
        assert_allclose(msd.odict[self.key2].err / msd.norm, self.err2)

    def test_get_sample(self):
        msd = self.get_msd()
        x = np.linspace(np.min((self.x1, self.x2)), np.max((self.x1, self.x2)), self.n*10)
        x2d = msd.sample(x)
        self.assertEqual(x2d.ndim, 2)
        assert_equal(x2d[:,0], np.r_[np.zeros_like(x), np.ones_like(x)])
        assert_equal(x2d[:,1], np.r_[x, x])

    def test_different_sized_arrays_from_items(self):
        items = [(self.key1, [self.x1, self.y1[:-1], self.err1]),
                 (self.key2, [self.x2, self.y2, self.err2])]
        with self.assertRaises(ValueError) as cm:
            util.data_from_items(items)
        self.assertEqual(str(cm.exception), '{} key has different array shapes'.format(self.key1))

    def test_different_sized_arrays_from_arrays(self):
        x_0 = np.r_[np.zeros(self.n), np.ones(self.n)]
        x_1 = np.r_[self.x1, self.x2]
        x = np.stack((x_0, x_1), axis=1)
        y = np.r_[self.y1, self.y2[:-1]]
        err = np.r_[self.err1, self.err2]

        with self.assertRaises(IndexError):
            util.data_from_arrays(x, y, err)

    def test_empty_data_from_items(self):
        items = [(self.key1, [[], [], []]), (self.key2, [[], [], []])]
        msd = util.data_from_items(items)
        assert_equal(msd.arrays.x, np.array([]).reshape(0,2))
        assert_equal(msd.arrays.y, np.array([]))
        assert_equal(msd.arrays.err, np.array([]))

    def test_empty_data_without_keys_from_arrays(self):
        msd = util.data_from_arrays(x=np.array([]).reshape(0,2), y=np.array([]), err=np.array([]))
        self.assertEqual(msd.odict, {})

    def test_empty_data_with_keys_from_arrays(self):
        msd = util.data_from_arrays(x=np.array([]).reshape(0,2), y=np.array([]), err=np.array([]),
                                    keys=[self.key1, self.key2])

        state_data = util.StateData(x=np.array([]), y=np.array([]), err=np.array([]))
        assert_equal(msd.odict[self.key1].x, state_data.x)
        assert_equal(msd.odict[self.key1].y, state_data.y)
        assert_equal(msd.odict[self.key1].err, state_data.err)
        assert_equal(msd.odict[self.key2].x, state_data.x)
        assert_equal(msd.odict[self.key2].y, state_data.y)
        assert_equal(msd.odict[self.key2].err, state_data.err)
        self.assertEqual(tuple(msd.keys()), (self.key1, self.key2))

    def test_append_same_keys(self):
        x1 = 1.
        y1 = 2.
        err1 = 3.
        items = [(self.key1, [np.array([x1]), np.array([y1]), np.array([err1])]),
                 (self.key2, [np.array([x1]), np.array([y1]), np.array([err1])])]
        msd1 = util.data_from_items(items)

        x2 = 4.
        y2 = 5.
        err2 = 6.
        items = [(self.key1, [np.array([x2]), np.array([y2]), np.array([err2])]),
                 (self.key2, [np.array([x2]), np.array([y2]), np.array([err2])])]
        msd2 = util.data_from_items(items)

        msd1.append(msd2)

        assert_equal(msd1.odict[self.key1].x, [x1, x2])
        assert_equal(msd1.odict[self.key1].y, [y1, y2])
        assert_equal(msd1.odict[self.key1].err, [err1, err2])
        assert_equal(msd1.odict[self.key2].x, [x1, x2])
        assert_equal(msd1.odict[self.key2].y, [y1, y2])
        assert_equal(msd1.odict[self.key2].err, [err1, err2])

        assert_equal(msd1.arrays.x[:, 0], [0, 0, 1, 1])
        assert_equal(msd1.arrays.x[:, 1], [x1, x2, x1, x2])
        assert_allclose(msd1.arrays.y * msd1.arrays.norm, [y1, y2, y1, y2])
        assert_allclose(msd1.arrays.err * msd1.arrays.norm, [err1, err2, err1, err2])

    def test_append_different_key(self):
        x1 = 1.
        y1 = 2.
        err1 = 3.
        items = [(self.key1, [np.array([x1]), np.array([y1]), np.array([err1])])]
        msd1 = util.data_from_items(items)

        x2 = 4.
        y2 = 5.
        err2 = 6.
        items = [(self.key2, [np.array([x2]), np.array([y2]), np.array([err2])])]
        msd2 = util.data_from_items(items)

        msd1.append(msd2)

        assert_equal(msd1.odict[self.key1].x, [x1])
        assert_equal(msd1.odict[self.key1].y, [y1])
        assert_equal(msd1.odict[self.key1].err, [err1])
        assert_equal(msd1.odict[self.key2].x, [x2])
        assert_equal(msd1.odict[self.key2].y, [y2])
        assert_equal(msd1.odict[self.key2].err, [err2])

        assert_equal(msd1.arrays.x[:, 0], [0, 1])
        assert_equal(msd1.arrays.x[:, 1], [x1, x2])
        assert_allclose(msd1.arrays.y * msd1.arrays.norm, [y1, y2])
        assert_allclose(msd1.arrays.err * msd1.arrays.norm, [err1, err2])

    def test_addition(self):
        msd1 = self.get_msd()
        msd2 = self.get_msd()
        msd = msd1 + msd2

        assert_equal(msd.odict[self.key1].x, self.x1)
        assert_equal(msd.odict[self.key1].y, self.y1*2)
        assert_allclose(msd.odict[self.key1].err, self.err1*np.sqrt(2))
        assert_equal(msd.odict[self.key2].x, self.x2)
        assert_equal(msd.odict[self.key2].y, self.y2*2)
        assert_allclose(msd.odict[self.key2].err, self.err2*np.sqrt(2))

        assert_equal(msd.arrays.x[:, 0], np.r_[np.zeros(self.n), np.ones(self.n)])
        assert_equal(msd.arrays.x[:, 1], np.r_[self.x1, self.x2])
        assert_allclose(msd.arrays.y * msd.arrays.norm, 2*np.r_[self.y1, self.y2])
        assert_allclose(msd.arrays.err * msd.arrays.norm, np.r_[self.err1, self.err2]*np.sqrt(2))

    def test_subtraction(self):
        msd1 = self.get_msd()
        msd2 = self.get_msd()
        msd = msd1 - msd2

        assert_equal(msd.odict[self.key1].x, self.x1)
        assert_equal(msd.odict[self.key1].y, 0)
        assert_allclose(msd.odict[self.key1].err, self.err1 * np.sqrt(2))
        assert_equal(msd.odict[self.key2].x, self.x2)
        assert_equal(msd.odict[self.key2].y, 0)
        assert_allclose(msd.odict[self.key2].err, self.err2 * np.sqrt(2))

        assert_equal(msd.arrays.x[:, 0], np.r_[np.zeros(self.n), np.ones(self.n)])
        assert_equal(msd.arrays.x[:, 1], np.r_[self.x1, self.x2])
        assert_allclose(msd.arrays.y * msd.arrays.norm, 0)
        assert_allclose(msd.arrays.err * msd.arrays.norm, np.r_[self.err1, self.err2] * np.sqrt(2))

    def test_multiplication(self):
        mul = 2
        msd = mul * self.get_msd()
        assert_equal(msd.odict[self.key1].x, self.x1)
        assert_equal(msd.odict[self.key1].y, mul*self.y1)
        assert_equal(msd.odict[self.key1].err, mul*self.err1)
        assert_equal(msd.odict[self.key2].x, self.x2)
        assert_equal(msd.odict[self.key2].y, mul*self.y2)
        assert_equal(msd.odict[self.key2].err, mul*self.err2)

        assert_equal(msd.arrays.x[:, 0], np.r_[np.zeros(self.n), np.ones(self.n)])
        assert_equal(msd.arrays.x[:, 1], np.r_[self.x1, self.x2])
        assert_allclose(msd.arrays.y * msd.arrays.norm, mul*np.r_[self.y1, self.y2])
        assert_allclose(msd.arrays.err * msd.arrays.norm, mul*np.r_[self.err1, self.err2])

    def test_addition_different_key_order(self):
        msd1 = self.get_msd()
        items2 = [(self.key2, [self.x2, self.y2, self.err2]),
                  (self.key1, [self.x1, self.y1, self.err1])]
        msd2 = util.data_from_items(items2)
        msd = msd1 - msd2

        assert_equal(msd.odict[self.key1].x, self.x1)
        assert_equal(msd.odict[self.key1].y, 0)
        assert_allclose(msd.odict[self.key1].err, self.err1 * np.sqrt(2))
        assert_equal(msd.odict[self.key2].x, self.x2)
        assert_equal(msd.odict[self.key2].y, 0)
        assert_allclose(msd.odict[self.key2].err, self.err2 * np.sqrt(2))

        assert_equal(msd.arrays.x[:, 0], np.r_[np.zeros(self.n), np.ones(self.n)])
        assert_equal(msd.arrays.x[:, 1], np.r_[self.x1, self.x2])
        assert_allclose(msd.arrays.y * msd.arrays.norm, 0)
        assert_allclose(msd.arrays.err * msd.arrays.norm, np.r_[self.err1, self.err2] * np.sqrt(2))

    def test_convert_arrays(self):
        msd1 = self.get_msd()
        x = msd1.arrays.x.copy()
        y = msd1.arrays.y * 2
        err = msd1.arrays.err / 2
        msd2 = msd1.convert_arrays(x, y, err)

        assert_equal(msd2.arrays.x, x)
        assert_equal(msd2.arrays.y, y)
        assert_equal(msd2.arrays.err, err)
        assert_equal(msd2.arrays.norm, msd1.arrays.norm)

        assert_equal(msd2.odict[self.key1].x, self.x1)
        assert_allclose(msd2.odict[self.key1].y, self.y1 * 2)
        assert_allclose(msd2.odict[self.key1].err, self.err1 / 2)
        assert_equal(msd2.odict[self.key2].x, self.x2)
        assert_allclose(msd2.odict[self.key2].y, self.y2 * 2)
        assert_allclose(msd2.odict[self.key2].err, self.err2 / 2)
