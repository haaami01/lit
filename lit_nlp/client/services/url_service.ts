/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {autorun} from 'mobx';

import {defaultValueByField, IndexedInput, Input, listFieldTypes, Spec} from '../lib/types';
import {isLitSubtype} from '../lib/utils';

import {LitService} from './lit_service';

/**
 * Interface for reading/storing app configuration from/to the URL.
 */
export class UrlConfiguration {
  selectedTab?: string;
  selectedModels: string[] = [];
  selectedData: string[] = [];
  primarySelectedData?: string;
  /**
   * For datapoints that are not in the original dataset, the fields
   * and their values are added directly into the url.
   * LIT can load multiple examples from the url, but can only share
   * primary selected example.
   */
  dataFields: {[key: number]: Input} = {};
  selectedDataset?: string;
  hiddenModules: string[] = [];
  compareExamplesEnabled?: boolean;
  layoutName?: string;
  /** Path to load a new dataset from, on pageload. */
  newDatasetPath?: string;
}

/**
 * Interface describing how AppState is synced to the URL Service
 */
export interface StateObservedByUrlService {
  currentModels: string[];
  currentDataset: string;
  setUrlConfiguration: (urlConfiguration: UrlConfiguration) => void;
  getUrlConfiguration: () => UrlConfiguration;
  currentDatasetSpec: Spec;
  compareExamplesEnabled: boolean;
  layoutName: string;
  getCurrentInputDataById: (id: string) => IndexedInput | null;
  indexDatapoints: (data: IndexedInput[]) => Promise<IndexedInput[]>;
  commitNewDatapoints: (datapoints: IndexedInput[]) => void;
}

/**
 * Interface describing how the ModulesService is synced to the URL Service
 */
export interface ModulesObservedByUrlService {
  hiddenModuleKeys: Set<string>;
  setUrlConfiguration: (urlConfiguration: UrlConfiguration) => void;
  selectedTab: string;
}

/**
 * Interface describing how the SelectionService is synced to the URL service
 */
export interface SelectionObservedByUrlService {
  readonly primarySelectedId: string|null;
  readonly primarySelectedInputData: IndexedInput|null;
  setPrimarySelection: (id: string) => void;
  readonly selectedIds: string[];
  selectIds: (ids: string[]) => void;
}

const SELECTED_TAB_KEY = 'tab';
const SELECTED_DATA_KEY = 'selection';
const PRIMARY_SELECTED_DATA_KEY = 'primary';
const SELECTED_DATASET_KEY = 'dataset';
const SELECTED_MODELS_KEY = 'models';
const HIDDEN_MODULES_KEY = 'hidden_modules';
const COMPARE_EXAMPLES_ENABLED_KEY = 'compare_data_mode';
const LAYOUT_KEY = 'layout';
const DATA_FIELDS_KEY_SUBSTRING = 'data';
/** Path to load a new dataset from, on pageload. */
const NEW_DATASET_PATH = 'new_dataset_path';

const MAX_IDS_IN_URL_SELECTION = 100;

const makeDataFieldKey = (key: string) => `${DATA_FIELDS_KEY_SUBSTRING}_${key}`;
const parseDataFieldKey = (key: string) => {
  const pieces = key.split('_');
  const indexStr = pieces[0].replace(`${DATA_FIELDS_KEY_SUBSTRING}`, '');
  return {fieldKey: pieces.slice(1).join('_'), dataIndex: +indexStr};
};

/**
 * Singleton service responsible for deserializing / serializing state to / from
 * a url.
 */
export class UrlService extends LitService {
  /** Parse arrays in a url param, filtering out empty strings */
  private urlParseArray(encoded: string) {
    if (encoded == null) {
      return [];
    }
    const array = encoded.split(',');
    return array.filter(str => str !== '');
  }

  /** Parse a string in a url param, filtering out empty strings */
  private urlParseString(encoded: string) {
    return encoded ? encoded : undefined;
  }

  /** Parse a boolean in a url param, if undefined return false */
  private urlParseBoolean(encoded: string) {
    return encoded === 'true';
  }

  /** Parse the data field based on its type */
  private parseDataFieldValue(fieldKey: string, encoded: string, spec: Spec) {
    const fieldSpec = spec[fieldKey];
    // If array type, unpack as an array.
    if (isLitSubtype(fieldSpec, listFieldTypes)) {
      return this.urlParseArray(encoded);
    } else {  // String-like.
      return this.urlParseString(encoded) ??
          defaultValueByField(fieldKey, spec);
    }
  }

  private getConfigurationFromUrl(): UrlConfiguration {
    const urlConfiguration = new UrlConfiguration();

    const urlSearchParams = new URLSearchParams(window.location.search);
    for (const [key, value] of Object.entries(urlSearchParams)) {
      if (key === SELECTED_MODELS_KEY) {
        urlConfiguration.selectedModels = this.urlParseArray(value);
      } else if (key === SELECTED_DATA_KEY) {
        urlConfiguration.selectedData = this.urlParseArray(value);
      } else if (key === PRIMARY_SELECTED_DATA_KEY) {
        urlConfiguration.primarySelectedData = this.urlParseString(value);
      } else if (key === SELECTED_DATASET_KEY) {
        urlConfiguration.selectedDataset = this.urlParseString(value);
      } else if (key === HIDDEN_MODULES_KEY) {
        urlConfiguration.hiddenModules = this.urlParseArray(value);
      } else if (key === COMPARE_EXAMPLES_ENABLED_KEY) {
        urlConfiguration.compareExamplesEnabled = this.urlParseBoolean(value);
      } else if (key === SELECTED_TAB_KEY) {
        urlConfiguration.selectedTab = this.urlParseString(value);
      } else if (key === LAYOUT_KEY) {
        urlConfiguration.layoutName = this.urlParseString(value);
      } else if (key === NEW_DATASET_PATH) {
        urlConfiguration.newDatasetPath = this.urlParseString(value);
      } else if (key.startsWith(DATA_FIELDS_KEY_SUBSTRING)) {
        const {fieldKey, dataIndex}: {fieldKey: string, dataIndex: number} =
            parseDataFieldKey(key);
        // TODO(b/179788207) Defer parsing of data keys here as we do not have
        // access to the input spec of the dataset at the time
        // this is called. We convert array fields to their proper forms in
        // syncSelectedDatapointToUrl.
        if (!(dataIndex in urlConfiguration.dataFields)) {
          urlConfiguration.dataFields[dataIndex] = {};
        }
        urlConfiguration.dataFields[dataIndex][fieldKey] = value;
      }
    }
    return urlConfiguration;
  }

  /** Set url parameter if it's not empty */
  private setUrlParam(
      params: URLSearchParams, key: string, data: string|string[]) {
    const value = data instanceof Array ? data.join(',') : data;
    if (value !== '' && value != null) {
      params.set(key, value);
    }
  }

  /**
   * If the datapoint was generated (not in the initial dataset),
   * set the data values themselves in the url.
   */
  setDataFieldURLParams(
      params: URLSearchParams, id: string,
      appState: StateObservedByUrlService) {
    const data = appState.getCurrentInputDataById(id);
    if (data !== null && data.meta['added']) {
      Object.keys(data.data).forEach((key: string) => {
        this.setUrlParam(params, makeDataFieldKey(key), data.data[key]);
      });
    }
  }

  /**
   * Parse the URL configuration and set it in the services that depend on it
   * for initializtion. Then, set up an autorun observer to automatically
   * react to changes of state and sync them to the url query params.
   */
  syncStateToUrl(
      appState: StateObservedByUrlService,
      selectionService: SelectionObservedByUrlService,
      modulesService: ModulesObservedByUrlService) {
    const urlConfiguration = this.getConfigurationFromUrl();
    appState.setUrlConfiguration(urlConfiguration);
    modulesService.setUrlConfiguration(urlConfiguration);

    const urlSelectedIds = urlConfiguration.selectedData || [];
    selectionService.selectIds(urlSelectedIds);

    // TODO(lit-dev) Add compared examples to URL parameters.
    // Only enable compare example mode if both selections and compare mode
    // exist in URL.
    if (selectionService.selectedIds.length > 0 &&
        urlConfiguration.compareExamplesEnabled) {
      appState.compareExamplesEnabled = true;
    }

    autorun(() => {
      const urlParams = new URLSearchParams();

      // Syncing app state
      this.setUrlParam(urlParams, SELECTED_MODELS_KEY, appState.currentModels);
      if (selectionService.selectedIds.length <= MAX_IDS_IN_URL_SELECTION) {
        this.setUrlParam(
            urlParams, SELECTED_DATA_KEY, selectionService.selectedIds);

        const id = selectionService.primarySelectedId;
        if (id != null) {
          this.setUrlParam(urlParams, PRIMARY_SELECTED_DATA_KEY, id);
          this.setDataFieldURLParams(urlParams, id, appState);
        }
      }
      this.setUrlParam(
          urlParams, SELECTED_DATASET_KEY, appState.currentDataset);

      this.setUrlParam(
          urlParams, COMPARE_EXAMPLES_ENABLED_KEY,
          appState.compareExamplesEnabled ? 'true' : 'false');

      // Syncing hidden modules
      this.setUrlParam(
          urlParams, HIDDEN_MODULES_KEY, [...modulesService.hiddenModuleKeys]);

      this.setUrlParam(urlParams, LAYOUT_KEY, appState.layoutName);
      this.setUrlParam(urlParams, SELECTED_TAB_KEY, modulesService.selectedTab);

      if (urlParams.toString() !== '') {
        const newUrl = `${window.location.pathname}?${urlParams.toString()}`;
        window.history.replaceState({}, '', newUrl);
      }
    });
  }


  /**
   * Syncing the selected datapoint in the URL is done separately from
   * the rest of the URL params. This is for when the selected
   * datapoint was not part of the original dataset: in this case, we
   * have to first load the dataset, and then create a new datapoint
   * from the fields stored in the url, and then select it.
   */
  async syncSelectedDatapointToUrl(
      appState: StateObservedByUrlService,
      selectionService: SelectionObservedByUrlService,
  ) {
    const urlConfiguration = appState.getUrlConfiguration();
    const dataFields = urlConfiguration.dataFields;
    const dataToAdd = Object.values(dataFields).map((fields: Input) => {
      // Create a new dict and do not modify the urlConfiguration. This makes
      // sure that this call works even if initialize app is called multiple
      // times.
      const outputFields: Input = {};
      const spec = appState.currentDatasetSpec;
      Object.keys(spec).forEach(key => {
        outputFields[key] = this.parseDataFieldValue(key, fields[key], spec);
      });
      const datum: IndexedInput = {
        data: outputFields,
        id: '',  // will be overwritten
        meta: {source: 'url', added: true},
      };
      return datum;
    });
    // If there are data fields set in the url, make new datapoints
    // from them.
    if (dataToAdd.length > 0) {
      const data = await appState.indexDatapoints(dataToAdd);
      appState.commitNewDatapoints(data);
      selectionService.selectIds(data.map((d) => d.id));
    }
    // Otherwise, use the primary selected datapoint url param directly.
    else {
      const id = urlConfiguration.primarySelectedData;
      if (id !== undefined) {
        selectionService.setPrimarySelection(id);
      }
    }
  }
}
