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

/* Importing the tensorflow library. */
import * as tf from '@tensorflow/tfjs';

/* Importing the `WebsitePhishingDataset` class from the `data.js` file. */
import { WebsitePhishingDataset } from './data';
import * as ui from './ui';
import * as utils from './utlis';

/**
 * It takes two tensors, `yTrue` and `yPred`, and returns the number of false positives
 * @param yTrue - The true labels
 * @param yPred - The predicted values
 * @returns The number of false positives.
 */
function flasePositive(yTrue, yPred) {
    return tf.tidy(() => {
        const one = tf.scalar(1);
        const zero = tf.scalar(0);
        return tf.logicalAnd(yTrue.equal(zero), yPred.equals(one)).sum().cast('float32')
    });
}

/**
 * It takes two tensors, `yTrue` and `yPred`, and returns the number of elements in `yTrue` that are
 * equal to 1 and the corresponding elements in `yPred` that are equal to 0
 * @param yTrue - The true labels
 * @param yPred - The predicted values
 * @returns The number of false negatives.
 */
function falseNegative(yTrue, yPred) {
    return tf.tidy(() => {
        const one = tf.scalar(1);
        const zero = tf.scalar(0);
        return tf.logicalAnd(yTrue.equal(one), yPred.equals(zero)).sum().cast('float32')
    });
}

/**
 * `truePositive` takes two tensors of the same shape and returns a scalar tensor with the number of
 * elements in the first tensor that are equal to 1 and the corresponding elements in the second tensor
 * are also equal to 1
 * @param yTrue - The true labels
 * @param yPred - The predicted values
 * @returns The number of true positives.
 */
function truePositive(yTrue, yPred) {
    return tf.tidy(() => {
        const one = tf.scalar(1);
        return tf.logicalAnd(yTrue.equal(one), yPred.equals(one)).sum().cast('float32')
    });
}

/**
 * `trueNegative` returns the number of times the model predicted a negative outcome and the actual
 * outcome was negative
 * @param yTrue - The true labels
 * @param yPred - The predicted values
 * @returns The number of true negatives.
 */
function trueNegative(yTrue, yPred) {
    return tf.tidy(() => {
        // const one = tf.scalar(1);
        const zero = tf.scalar(0);
        return tf.logicalAnd(yTrue.equal(zero), yPred.equals(zero)).sum().cast('float32')
    });
}

/**
 * `falsePositiveRate` is a function that takes two tensors as arguments and returns a tensor. The
 * function is defined using the `tf.tidy` function. The `tf.tidy` function takes a function as an
 * argument and returns a tensor. The function passed to `tf.tidy` takes no arguments and returns a
 * tensor. The function passed to `tf.tidy` returns the result of dividing the result of calling the
 * `flasePositive` function by the result of adding the result of calling the `flasePositive` function
 * to the result of calling the `trueNegative` function
 * @param yTrue - The true labels.
 * @param yPred - The predicted values
 * @returns The false positive rate is the ratio of false positives (FP) to the sum of false positives
 * (FP) and true negatives (TN).
 */

// TODO(cals): Use tf.metrics.falsePositiveRate when available.
function falsePositiveRate(yTrue, yPred) {
    return tf.tidy(() => {
        const fp = flasePositive(yTrue, yPred);
        const tn = trueNegative(yTrue, yPred);
        return fp.div(fp.add(tn));
    });
}

/**
 * `falseNegativeRate` is a function that takes two tensors as arguments, `yTrue` and `yPred`, and
 * returns a tensor that is the ratio of the number of false negatives to the total number of actual
 * positives
 * @param yTrue - The ground truth values.
 * @param yPred - The predicted values
 * @returns The false negative rate is the number of false negatives divided by the total number of
 * negatives.
 */

// TODO(cals): Use tf.metrics.falseNegativeRate when available.
function falseNegativeRate(yTrue, yPred) {
    return tf.tidy(() => {
        const fn = falseNegative(yTrue, yPred);
        const tp = truePositive(yTrue, yPred);
        return fn.div(fn.add(tp));
    });
}

/**
 * It takes the targets and the predictions, and for each threshold, it calculates the true positive
 * rate and the false positive rate
 * @param targets - The actual labels of the data.
 * @param probs - The probabilities of the predictions.
 * @param epoch - The current epoch number.
 * @returns The area under the curve.
 */
function drawROC(targets, probs, epoch) {
    return tf.tidy(() => {
        const thresholds = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0
        ]
        /* `tprs` is an array that will hold the true positive rates. */
        const tprs = []
        /* `fprs` is an array that will hold the false positive rates. */
        const fprs = []
        let area = 0

        for (let i = 0; i < thresholds.length; ++i) {
            const threshold = thresholds[i];

            const threshPredictions = utils.binarize(probs, threshold).as1D();
            const fpr = falsePositiveRate(targets, threshPredictions).dataSync()[0];
            const tpr = tf.metrics.recall(targets, threshPredictions).dataSync()[0];
            fprs.push(fpr);
            tprs.push(tpr);

            // Accumulate to area for AUC calculation.
            if (i > 0) {
                area += (tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i]) / 2;
            }
        }
        ui.plotROC(fprs, tprs, epoch);
        return area;
    });
}

const epochs = 400;
const batchSize = 350;

/* Creating a new instance of the `WebsitePhishingDataset` class. */
const data = new WebsitePhishingDataset();

data.loadData().then(async () => {
    await ui.updateStatus('Getting training and testing data...');
    const trainData = data.getTrainData();
    const testData = data.getTestData();

    await ui.updateStatus('Building model...');
    const model = tf.sequential();
    model.add(tf.layers.dense(
        { inputShape: [data.numFeatures], units: 100, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 100, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile(
        { optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    const trainLogs = [];
    let auc;

    await ui.updateStatus('Training starting...');
    /* Training the model. */
    await model.fit(trainData.data, trainData.target, {
        batchSize,
        epochs,
        validationSplit: 0.2,
        callbacks: {
            onEpochBegin: async (epoch) => {
                // Draw ROC every a few epochs.
                if ((epoch + 1) % 100 === 0 || epoch === 0 || epoch === 2 ||
                    epoch === 4) {
                    const probs = model.predict(testData.data);
                    auc = drawROC(testData.target, probs, epoch);
                }
            },
            onEpochEnd: async (epoch, logs) => {
                await ui.updateStatus(`Epoch ${epoch + 1} of ${epochs} completed.`);
                trainLogs.push(logs);
                ui.plotLosses(trainLogs);
                ui.plotAccuracies(trainLogs);
            }
        }
    });

    await ui.updateStatus('Running on test data...');
    /* Evaluating the model on the test data. */
    tf.tidy(() => {
        const result =
            model.evaluate(testData.data, testData.target, { batchSize: batchSize });

        const lastTrainLog = trainLogs[trainLogs.length - 1];
        const testLoss = result[0].dataSync()[0];
        const testAcc = result[1].dataSync()[0];

        const probs = model.predict(testData.data);
        const predictions = utils.binarize(probs).as1D();

        const precision =
            tf.metrics.precision(testData.target, predictions).dataSync()[0];
        const recall =
            tf.metrics.recall(testData.target, predictions).dataSync()[0];
        const fpr = falsePositiveRate(testData.target, predictions).dataSync()[0];

        /* Updating the status of the model. */
        ui.updateStatus(
            `Final train-set loss: ${lastTrainLog.loss.toFixed(4)} accuracy: ${lastTrainLog.acc.toFixed(4)}\n` +
            `Final validation-set loss: ${lastTrainLog.val_loss.toFixed(
                4)} accuracy: ${lastTrainLog.val_acc.toFixed(4)}\n` +
            `Test-set loss: ${testLoss.toFixed(4)} accuracy: ${testAcc.toFixed(4)}\n` +
            `Precision: ${precision.toFixed(4)}\n` +
            `Recall: ${recall.toFixed(4)}\n` +
            `False positive rate (FPR): ${fpr.toFixed(4)}\n` +
            `Area under the curve (AUC): ${auc.toFixed(4)}`);
    });
});