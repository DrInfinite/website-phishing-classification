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

/* Importing the tensorflow visualization library. */
import * as tfvis from '@tensorflow/tfjs-vis';

/* Getting the element with the id 'status' from the HTML file. */
const statusElement = document.getElementById('status');
/**
 * It takes a string as an argument and updates the status element with the string
 * @param message - The message to display in the status element.
 */
export function updateStatus(message) {
    statusElement.innerText = message;
};

/**
 * It takes the training logs and plots the loss and validation loss
 * @param trainLogs - The training logs returned by the fit() function.
 * @returns A promise.
 */
export async function plotLosses(trainLogs) {
    return tfvis.show.history(
        document.getElementById('plotLoss'), trainLogs, ['loss', 'val_loss'], {
        width: 450,
        height: 320,
        xLabel: 'Epoch',
        yLabel: 'Loss',
    });
}

/**
 * It takes the training logs and plots the accuracy and validation accuracy
 * @param trainLogs - The training logs returned by the model.fit() function.
 */
export async function plotAccuracies(trainLogs) {
    tfvis.show.history(
        document.getElementById('plotAccuracy'), trainLogs, ['acc', 'val_acc'], {
        width: 450,
        height: 320,
        xLabel: 'Epoch',
        yLabel: 'Accuracy',
    });
}

const rocValues = [];
const rocSeries = [];

/**
 * It takes the false positive rate and true positive rate arrays, and plots them on a line chart
 * @param fprs - false positive rates
 * @param tprs - true positive rates
 * @param epoch - The current epoch number.
 * @returns The return value is a promise.
 */
export async function plotROC(fprs, tprs, epoch) {
    epoch++;

    /* Adding the epoch number to the rocSeries array. */
    const seriesName = 'epoch ' +
        (epoch < 10 ? `00${epoch}` : (epoch < 100 ? `0${epoch}` : `${epoch}`))
    rocSeries.push(seriesName);

    const newSeries = [];
    /* Adding the false positive rate and true positive rate values to the newSeries array. */
    for (let i = 0; i < fprs.length; i++) {
        newSeries.push({
            x: fprs[i],
            y: tprs[i],
        });
    }
    rocValues.push(newSeries);

    /* Plotting the ROC curve. */
    return tfvis.render.linechart(
        document.getElementById('rocCurve'),
        { values: rocValues, series: rocSeries },
        {
            width: 450,
            height: 320,
        },
    );
}