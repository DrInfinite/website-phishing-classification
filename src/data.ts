/**
 * @license
 *
 * Copyright 2022 [T L Naparajith]. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs'
import * as utlis from './utlis'

const TRAIN_DATA = 'train-data';
const TRAIN_TARGET = 'train-target';
const TEST_DATA = 'test-data';
const TEST_TARGET = 'test-target';

/* It loads the data, normalizes it, and returns it as tensors */
export class WebsitePhishingDataset {
    /* Defining the variables. */
    dataset: [any, any, any, any];
    trainSize: number;
    testSize: number;
    trainBatchIndex: number;
    testBatchIndex: number;
    NUM_FEATURES: number;
    NUM_CLASSES: number;
    /**
     * This function is used to create a new instance of the class DataSet.
     */
    constructor() {
        this.dataset = null;
        this.trainSize = 0;
        this.testSize = 0;
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;

        this.NUM_FEATURES = 30;
        this.NUM_CLASSES = 2;
    }

    /**
     * This function returns the number of features in the dataset.
     * @returns The number of features.
     */
    get numFeatures() {
        return this.NUM_FEATURES;
    }

    /**
     * "This function loads the data from the csv files and normalizes it."
     *
     * The first line of the function is an async function. This means that the function will return a
     * promise. This is useful because we want to load the data from the csv files before we start
     * training the model.
     *
     * The next line is a bit more complicated. We are using the Promise.all() function to load the
     * data from the csv files. This function takes an array of promises and returns a promise that
     * resolves when all of the promises in the array have resolved.
     *
     * The next line is a bit confusing. We are using the normalizeDataset() function to normalize the
     * data. This function takes the data from the csv files and normalizes it. The normalizeDataset()
     * function returns an object with two properties: dataset and vectorMeans. The dataset property is
     * the normalized data and the vectorMe
     */
    async loadData() {
        this.dataset = await Promise.all([
            utlis.loadCsv(TRAIN_DATA), utlis.loadCsv(TRAIN_TARGET),
            utlis.loadCsv(TEST_DATA), utlis.loadCsv(TEST_TARGET)
        ]);

        let { dataset: trainDataset, vectorMeans, vectorStddevs } =
            utlis.normalizeDataset(this.dataset[0]);

        this.dataset[0] = trainDataset;

        let { dataset: testDataset } = utlis.normalizeDataset(
            this.dataset[2], false, vectorMeans, vectorStddevs);

        this.dataset[2] = testDataset;

        this.trainSize = this.dataset[0].length;
        this.testSize = this.dataset[2].length;

        utlis.shuffle(this.dataset[0], this.dataset[1]);
        utlis.shuffle(this.dataset[2], this.dataset[3]);
    }

    /**
     * "The function takes the first two arrays from the dataset array and flattens them into a single
     * array, then returns them as a tensor."
     *
     * The function is called in the following way:
     *
     * const { data, target } = this.getTrainData();
     *
     * The function returns an object with two properties, data and target.
     *
     * The data property is a tensor with the shape [this.trainSize, this.NUM_FEATURES].
     *
     * The target property is a tensor with the shape [this.trainSize].
     *
     * The trainSize property is the number of rows in the dataset array.
     *
     * The NUM_FEATURES property is the number of columns in the dataset array.
     *
     * The dataset array is a two dimensional array.
     *
     * The first row of the dataset array is the data.
     *
     * The second
     * @returns The data and target tensors.
     */
    getTrainData() {
        const dataShape = [this.trainSize, this.NUM_FEATURES];

        const trainData = Float32Array.from([].concat.apply([], this.dataset[0]));
        const trainTarget = Float32Array.from([].concat.apply([], this.dataset[1]));

        return {
            data: tf.tensor2d(trainData, dataShape),
            target: tf.tensor1d(trainTarget)
        };
    }

    /**
     * It takes the test data and target data from the dataset and converts it into a tensor2d and
     * tensor1d respectively.
     * @returns The data and target tensors are being returned.
     */
    getTestData() {
        const dataShape = [this.testSize, this.NUM_FEATURES];

        const testData = Float32Array.from([].concat.apply([], this.dataset[2]));
        const testTarget = Float32Array.from([].concat.apply([], this.dataset[3]));

        return {
            data: tf.tensor2d(testData, dataShape),
            target: tf.tensor1d(testTarget)
        };
    }
}