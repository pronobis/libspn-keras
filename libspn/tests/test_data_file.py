#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
from unittest.mock import patch
import tensorflow as tf


class TestFileDataset(TestCase):

    def test_get_files_labels(self):
        """Obtaining files/labels list from a specification in FileDataset"""
        def run(paths, true_files, true_labels):
            with self.subTest(paths=paths):
                files, labels = spn.FileDataset._get_files_labels(
                    self.data_path(paths))
                self.assertEqual(files, self.data_path(true_files))
                self.assertEqual(labels, true_labels)

        # Single path without glob and label
        run('img_dir1/img1-A.png',
            ['img_dir1/img1-A.png'],
            [''])

        # Single path with glob and no label
        run('img_dir*/img*-*.png',
            ['img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png',
             'img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir3/img1-A.png',
             'img_dir3/img2-C.png',
             'img_dir3/img3-B.png',
             'img_dir4/img1-A.png',
             'img_dir4/img2-C.png',
             'img_dir4/img3-B.png'],
            ['', '', '', '', '', '', '', '', '', '', '', ''])

        # Single path with a glob and a label
        run('img_dir*/img*-{*}.png',
            ['img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png',
             'img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir3/img1-A.png',
             'img_dir3/img2-C.png',
             'img_dir3/img3-B.png',
             'img_dir4/img1-A.png',
             'img_dir4/img2-C.png',
             'img_dir4/img3-B.png'],
            ['A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B'])

        # Multiple paths without glob and label
        # Tests if order of files is preserved
        run(['img_dir2/img2-C.png',
             'img_dir1/img1-A.png'],
            ['img_dir2/img2-C.png',
             'img_dir1/img1-A.png'],
            ['', ''])

        # Multiple paths with glob and no label
        # Tests if order of file specs is preserved
        run(['img_dir2/img*-*.png',
             'img_dir1/img*-*.png'],
            ['img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png'],
            ['', '', '', '', '', ''])

        # Multiple paths with a glob and a label
        run(['img_dir2/img*-{*}.png',
             'img_dir1/img*-{*}.png'],
            ['img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png'],
            ['A', 'C', 'B', 'A', 'C', 'B'])

    def test_get_files_labels_classes(self):
        """Obtaining files/labels list from a specification in FileDataset
        for specific class labels."""
        def run(paths, classes, true_files, true_labels):
            with self.subTest(paths=paths, classes=classes):
                files, labels = spn.FileDataset._get_files_labels(
                    self.data_path(paths), classes)
                self.assertEqual(files, self.data_path(true_files))
                self.assertEqual(labels, true_labels)

        # Single path without glob and label
        run('img_dir1/img1-A.png',
            ['A'],
            [], [])

        # Single path with glob and no label
        run('img_dir*/img*-*.png',
            ['A'],
            [], [])

        # Single path with a glob and a label
        run('img_dir*/img*-{*}.png',
            ['C'],
            ['img_dir1/img2-C.png',
             'img_dir2/img2-C.png',
             'img_dir3/img2-C.png',
             'img_dir4/img2-C.png'],
            ['C', 'C', 'C', 'C'])
        run('img_dir*/img*-{*}.png',
            ['B', 'A'],
            ['img_dir1/img1-A.png',
             'img_dir1/img3-B.png',
             'img_dir2/img1-A.png',
             'img_dir2/img3-B.png',
             'img_dir3/img1-A.png',
             'img_dir3/img3-B.png',
             'img_dir4/img1-A.png',
             'img_dir4/img3-B.png'],
            ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])

        # Multiple paths without glob and label
        # Tests if order of files is preserved
        run(['img_dir2/img2-C.png',
             'img_dir1/img1-A.png'],
            ['A'],
            [], [])

        # Multiple paths with glob and no label
        # Tests if order of file specs is preserved
        run(['img_dir2/img*-*.png',
             'img_dir1/img*-*.png'],
            ['A'],
            [], [])

        # Multiple paths with a glob and a label
        run(['img_dir2/img*-{*}.png',
             'img_dir1/img*-{*}.png'],
            ['B'],
            ['img_dir2/img3-B.png',
             'img_dir1/img3-B.png'],
            ['B', 'B'])
        run(['img_dir2/img*-{*}.png',
             'img_dir1/img*-{*}.png'],
            ['C', 'A'],
            ['img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir1/img1-A.png',
             'img_dir1/img2-C.png'],
            ['A', 'C', 'A', 'C'])

    @patch.multiple(spn.FileDataset, __abstractmethods__=set())  # Make Dataset non-abstract
    def test_file_label_serving(self):
        """Serving files and labels using FileDataset"""
        def run(paths, true_files, true_labels):
            with self.subTest(paths=paths):
                dataset = spn.FileDataset(self.data_path(paths),
                                          num_vars=1, num_vals=1,
                                          num_labels=1,  # Not relevant here
                                          num_epochs=1, batch_size=1,
                                          shuffle=False, shuffle_batch=False)
                fqueue = dataset._get_file_queue()
                ftensor, ltensor = dataset._get_file_label_tensors()
                files1 = []
                files2 = []
                labels = []
                with spn.session() as (sess, run):
                    while run():
                        f = sess.run(fqueue.dequeue())
                        files1.append(str(f, 'utf-8'))
                with spn.session() as (sess, run):
                    while run():
                        f, l = sess.run([ftensor, ltensor])
                        files2.append(str(f, 'utf-8'))
                        labels.append(str(l, 'utf-8'))
                self.assertEqual(files1, self.data_path(true_files))
                self.assertEqual(files2, self.data_path(true_files))
                self.assertEqual(labels, true_labels)

        # Single path without glob and label
        run('img_dir1/img1-A.png',
            ['img_dir1/img1-A.png'],
            [''])

        # Single path with glob and no label
        run('img_dir*/img*-*.png',
            ['img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png',
             'img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir3/img1-A.png',
             'img_dir3/img2-C.png',
             'img_dir3/img3-B.png',
             'img_dir4/img1-A.png',
             'img_dir4/img2-C.png',
             'img_dir4/img3-B.png'],
            ['', '', '', '', '', '', '', '', '', '', '', ''])

        # Single path with a glob and a label
        run('img_dir*/img*-{*}.png',
            ['img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png',
             'img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir3/img1-A.png',
             'img_dir3/img2-C.png',
             'img_dir3/img3-B.png',
             'img_dir4/img1-A.png',
             'img_dir4/img2-C.png',
             'img_dir4/img3-B.png'],
            ['A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B'])

        # Multiple paths without glob and label
        # Tests if order of files is preserved
        run(['img_dir2/img2-C.png',
             'img_dir1/img1-A.png'],
            ['img_dir2/img2-C.png',
             'img_dir1/img1-A.png'],
            ['', ''])

        # Multiple paths with glob and no label
        # Tests if order of file specs is preserved
        run(['img_dir2/img*-*.png',
             'img_dir1/img*-*.png'],
            ['img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png'],
            ['', '', '', '', '', ''])

        # Multiple paths with a glob and a label
        run(['img_dir2/img*-{*}.png',
             'img_dir1/img*-{*}.png'],
            ['img_dir2/img1-A.png',
             'img_dir2/img2-C.png',
             'img_dir2/img3-B.png',
             'img_dir1/img1-A.png',
             'img_dir1/img2-C.png',
             'img_dir1/img3-B.png'],
            ['A', 'C', 'B', 'A', 'C', 'B'])


if __name__ == '__main__':
    tf.test.main()
