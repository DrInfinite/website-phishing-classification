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

/* Importing the libraries. */
import * as tf from '@tensorflow/tfjs'
import * as Papa from 'papaparse'

const BASE_URL =
    'https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/';

/**
 * It takes an array of objects, sorts the keys, and returns an array of arrays.
 * @param data - the data from the csv file
 * @returns An array of arrays.
 */
async function parseCsv(data) {
    return new Promise(resolve => {
        data = data.map((row) => {
            return Object.keys(row).sort().map(key => parseFloat(row[key]));
        });
        resolve(data);
    });
};

/**
 * It downloads a CSV file from a URL, parses it, and returns the parsed data as a promise.
 *
 * The function takes a single argument, the name of the CSV file to download.
 *
 * The function returns a promise that resolves to the parsed CSV data.
 *
 * The function uses the Papa Parse library to download and parse the CSV file.
 *
 * The function uses the parseCsv() function to parse the CSV data.
 *
 * The function uses the BASE_URL constant to build the URL of the CSV file to download.
 *
 * The function uses the Papa Parse library to download and parse the CSV file.
 *
 * The function uses the parseCsv() function to parse the CSV data.
 *
 * The function uses the BASE_URL constant to build the URL of the CSV file to download.
 *
 * The function uses the Papa Parse library
 * @param filename - The name of the file to load.
 * @returns A promise that resolves to an array of objects.
 */
export async function loadCsv(filename) {
    return new Promise(resolve => {
        const url = `${BASE_URL}${filename}.csv`;

        console.log(`  * Downloading data from: ${url}`);
        Papa.parse(url, {
            download: true,
            header: true,
            complete: (results) => {
                resolve(parseCsv(results['data']));
            }
        })
    });
};

/**
 * It shuffles the data and label arrays in the same way
 * @param data - the data to be shuffled
 * @param label - the label of the data, which is a one-hot vector.
 */
export async function shuffle(data, label) {
    let counter = data.length;
    let temp = 0;
    let index = 0;
    /* Shuffling the data and label arrays in the same way. */
    while (counter > 0) {
        index = (Math.random() * counter) | 0;
        counter--;
        // data:
        temp = data[counter];
        data[counter] = data[index];
        data[index] = temp;
        // label:
        temp = label[counter];
        label[counter] = label[index];
        label[index] = temp;
    }
};

/**
 * It takes a vector of numbers and returns the mean of those numbers
 * @param vector - the vector to calculate the mean of
 * @returns The mean of the vector.
 */
function mean(vector) {
    let sum = 0;
    for (const x of vector) {
        sum += x;
    }
    return sum / vector.length;
};

/**
 * It calculates the standard deviation of a vector by subtracting the mean from each element, squaring
 * the result, summing the squares, dividing by the number of elements minus one, and taking the square
 * root
 * @param vector - the array of numbers you want to find the standard deviation of
 * @returns The standard deviation of the vector.
 */
function stddev(vector) {
    let squareSum = 0;
    const vectorMean = mean(vector);
    for (const x of vector) {
        squareSum += (x - vectorMean) * (x - vectorMean);
    }
    return Math.sqrt(squareSum / (vector.length - 1));
};

/**
 * It takes a vector, and returns a new vector where each element is the difference between the
 * original element and the mean of the original vector, divided by the standard deviation of the
 * original vector
 * @param vector - the vector to normalize
 * @param vectorMean - The mean of the vector.
 * @param vectorStddev - The standard deviation of the vector.
 * @returns a new array with the normalized values.
 */
const normalizeVector = (vector, vectorMean, vectorStddev) => {
    return vector.map(x => (x - vectorMean) / vectorStddev);
};

/**
 * It takes a dataset, and returns a normalized version of the dataset, along with the mean and
 * standard deviation of each feature
 * @param dataset - The dataset to normalize.
 * @param [isTrainData=true] - A boolean value that indicates whether the dataset is the training
 * data or not.
 * @param [vectorMeans] - The mean of each feature vector.
 * @param [vectorStddevs] - The standard deviation of each feature vector.
 * @returns an object with three properties: dataset, vectorMeans, and vectorStddevs.
 */
export function normalizeDataset(
    dataset, isTrainData = true, vectorMeans = [], vectorStddevs = []) {
    const numFeatures = dataset[0].length;
    let vectorMean;
    let vectorStddev;

    /* Normalizing the data. */
    for (let i = 0; i < numFeatures; i++) {
        const vector = dataset.map(row => row[i]);

        /* Calculating the mean and standard deviation of the training data, and then using those
        values to normalize the test data. */
        if (isTrainData) {
            vectorMean = mean(vector);
            vectorStddev = stddev(vector);

            vectorMeans.push(vectorMean);
            vectorStddevs.push(vectorStddev);
        } else {
            vectorMean = vectorMeans[i];
            vectorStddev = vectorStddevs[i];
        }

        const vectorNormalized =
            normalizeVector(vector, vectorMean, vectorStddev);

        /* Iterating over the normalized vector and assigning the normalized value to the dataset. */
        vectorNormalized.forEach((value, index) => {
            dataset[index][i] = value;
        });
    }

    return { dataset, vectorMeans, vectorStddevs };
};

/**
 * If the value of y is greater than the threshold, return 1, otherwise return 0.
 * @param y - The output of the model.
 * @param threshold - The threshold value above which the prediction is considered positive.
 * @returns A tensor of the same shape as y, where values greater than threshold are 1 and values less
 * than threshold are 0.
 */
export function binarize(y, threshold) {
    if (threshold == null) {
        threshold = 0.5;
    }
    tf.util.assert(
        threshold >= 0 && threshold <= 1,
        `Expected threshold to be >=0 and <=1, but got ${threshold}`);

    return tf.tidy(() => {
        const condition = y.greater(tf.scalar(threshold));
        return tf.where(condition, tf.onesLike(y), tf.zerosLike(y));
    });
}
