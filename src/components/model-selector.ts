/**
 * Copyright 2026 The MediaPipe Authors.
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

import { ViewToggle } from './view-toggle';
import { formatMegabytes, isModelCached, removeCachedModel } from './model-cache';

export interface ModelOption {
  label: string;
  value: string;
  isDefault?: boolean;
  /** Listed so users know it is coming, but not selectable yet. */
  disabled?: boolean;
}

export type ModelSelection = { type: 'standard'; value: string } | { type: 'custom'; file: File };

export interface ModelSelectorConfig {
  /**
   * When true (default) a standard model is loaded as soon as it is picked in
   * the dropdown. When false a "Initialize Task" button is shown and nothing is
   * downloaded until the user presses it; use this for large models.
   */
  autoLoad?: boolean;
  /** Accepted file extensions for the Upload tab. */
  accept?: string;
  /** Label of the upload button. */
  uploadLabel?: string;
  /**
   * Maps a standard option to the URL it is downloaded from. Models that are
   * already in the local cache are marked with a ✓ in the dropdown.
   */
  resolveUrl?: (value: string) => string | undefined;
  /**
   * Explicit-load selectors only: if the default model has not been downloaded
   * yet but another one has (e.g. by a different demo), pre-select that one so
   * "Initialize Task" is instant. Never changes a selection the user already made.
   */
  preferCached?: boolean;
}

const DEFAULT_CONFIG = {
  autoLoad: true,
  accept: '.tflite,.task',
  uploadLabel: 'Choose .tflite File',
};

/**
 * Fallback tags appended to dropdown labels in browsers whose `<select>` can
 * only show plain text. Where the customizable select is available (Chrome
 * 135+, `appearance: base-select`) the list shows a graphic pill instead (see
 * `.model-select option::after` in app_clean.css) and the control always shows
 * a badge for the selected model.
 */
const DOWNLOADED_TAG = '[loaded]';
const LOADED_TAG = '[active]';
const RICH_OPTIONS = typeof CSS !== 'undefined' && CSS.supports('appearance', 'base-select');

/**
 * Model picker shared by every task: a "Standard" tab with the built-in models
 * and an "Upload" tab for local files, plus a download progress bar.
 *
 * Standard models are either loaded immediately on selection (`autoLoad`, the
 * default for small models) or only after the user presses "Initialize Task"
 * (`autoLoad: false`, for large models that should not be downloaded until
 * explicitly requested). Uploads always load right away.
 */
export class ModelSelector {
  private container: HTMLElement;
  private options: ModelOption[];
  private onModelChanged: (selection: ModelSelection) => void | Promise<void>;
  private config: typeof DEFAULT_CONFIG & ModelSelectorConfig;
  private currentMode: 'standard' | 'upload' = 'standard';
  private lastSelection: ModelSelection | null = null;
  private busy = false;
  /** True while the task owns the standard status text (loading / loaded / failed). */
  private statusPinned = false;
  /** Set once the user (or the task) picked a model, so `preferCached` leaves it alone. */
  private selectionFixed = false;
  /** Lets a newer cache check invalidate the result of an older, still-running one. */
  private cacheCheckGeneration = 0;
  /** Standard models known to be in the local cache (from the last check). */
  private cachedValues = new Set<string>();
  /** Standard model currently loaded in memory by the task, if any. */
  private loadedValue: string | null = null;
  /**
   * Explicit-load selectors: armed on tab open and on each dropdown change, so
   * the next cache check loads the selected model if it is already on disk.
   */
  private autoLoadPending = true;

  private modelSelect!: HTMLSelectElement;
  private badgeContainer!: HTMLElement;
  private badge!: HTMLElement;
  private deleteBtn!: HTMLButtonElement;
  private loadButton: HTMLButtonElement | null = null;
  private standardStatus: HTMLElement | null = null;
  private modelUpload!: HTMLInputElement;
  private uploadStatus!: HTMLElement;
  private progressWrap!: HTMLElement;
  private progressBar!: HTMLElement;
  private progressText!: HTMLElement;

  constructor(
    containerId: string,
    options: ModelOption[],
    onModelChanged: (selection: ModelSelection) => void | Promise<void>,
    config: ModelSelectorConfig = {}
  ) {
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`ModelSelector: container ${containerId} not found`);
    this.container = el;
    this.options = options;
    this.onModelChanged = onModelChanged;
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.render();
  }

  public updateOptions(newOptions: ModelOption[]) {
    this.options = newOptions;
    this.renderOptions();
    this.refreshCacheState();
  }

  /** The value of the standard model currently picked in the dropdown. */
  public getSelectedValue(): string {
    return this.modelSelect.value;
  }

  /** Selects a standard model in the dropdown without loading it. */
  public setSelectedValue(value: string) {
    if (this.options.some((opt) => opt.value === value)) {
      this.modelSelect.value = value;
      this.selectionFixed = true;
      this.refreshCacheState();
    }
  }

  /** The last selection the user asked to load, if any. */
  public getLastSelection(): ModelSelection | null {
    return this.lastSelection;
  }

  /**
   * Disables the controls while a model is loading. Only applies to selectors
   * with an explicit "Initialize Task" button: auto-loading tasks keep the dropdown
   * live so a different model can be picked at any time.
   */
  public setBusy(busy: boolean) {
    this.busy = busy;
    if (this.config.autoLoad) return;
    this.modelSelect.disabled = busy;
    this.modelUpload.disabled = busy;
    if (this.loadButton) {
      this.loadButton.disabled = busy;
      this.loadButton.textContent = busy ? 'Initializing…' : 'Initialize Task';
    }
    this.updateLoadButton();
  }

  /** Sets the status line under the source (Standard or Upload) that was loaded last. */
  public setStatus(text: string) {
    const target = this.lastSelection?.type === 'custom' ? this.uploadStatus : this.standardStatus;
    if (!target) return;
    target.textContent = text;
    // The task now owns the standard status; keep the cache hint from replacing it.
    if (target === this.standardStatus) this.statusPinned = true;
  }

  /**
   * Tells the selector which standard model is now loaded in memory (`null`
   * for none / an uploaded file), which switches its badge to "Loaded".
   */
  public setLoaded(value: string | null) {
    this.loadedValue = value;
    this.renderModelMarks();
  }

  /**
   * Re-checks which standard models are already in the local cache: tags them
   * in the dropdown, optionally pre-selects one (`preferCached`), loads the
   * selected one if it is already on disk (the "Initialize Task" button only gates
   * downloads) and otherwise updates the hint under the button (unless the task
   * has set its own status for the selected model).
   */
  public async refreshCacheState() {
    if (!this.config.resolveUrl) return;
    const resolveUrl = this.config.resolveUrl;
    const generation = ++this.cacheCheckGeneration;

    const cached = new Set<string>();
    await Promise.all(
      this.options.map(async (opt) => {
        const url = resolveUrl(opt.value);
        if (url && (await isModelCached(url))) cached.add(opt.value);
      })
    );
    // A newer check superseded this one (options changed, model removed, ...).
    if (generation !== this.cacheCheckGeneration) return;
    this.cachedValues = cached;

    // Prefer a model that is already on disk over one that still needs a download.
    if (this.config.preferCached && !this.config.autoLoad && !this.selectionFixed && !this.statusPinned) {
      if (!cached.has(this.modelSelect.value)) {
        const preferred = this.options.find((opt) => cached.has(opt.value) && !opt.disabled);
        if (preferred) this.modelSelect.value = preferred.value;
      }
    }

    this.renderModelMarks();

    // Already downloaded: nothing to confirm, so initialize right away. Only on
    // tab open / dropdown change – never behind the user's back after an upload,
    // a failed load or while another load is running.
    const value = this.modelSelect.value;
    const autoLoad = this.autoLoadPending && !this.config.autoLoad;
    this.autoLoadPending = false;
    if (autoLoad && cached.has(value) && !this.busy && !this.statusPinned && this.loadedValue !== value) {
      this.emit({ type: 'standard', value });
      return;
    }

    if (!this.standardStatus || this.statusPinned || this.busy) return;
    this.renderCacheState(resolveUrl(value), cached.has(value));
  }

  public showProgress(loaded: number, total: number) {
    this.progressWrap.style.display = 'block';
    if (total > 0) {
      const percent = Math.min(100, Math.round((loaded / total) * 100));
      this.progressBar.style.width = `${percent}%`;
      this.progressText.innerText = `Loading Model... ${percent}% (${formatMegabytes(loaded)} / ${formatMegabytes(total)})`;
    } else {
      this.progressBar.style.width = '100%';
      this.progressText.innerText = `Loading Model... ${formatMegabytes(loaded)}`;
    }
  }

  public hideProgress() {
    this.progressWrap.style.display = 'none';
    this.progressBar.style.width = '0%';
  }

  private render() {
    const id = this.container.id;
    const explicitLoad = !this.config.autoLoad;

    // 1. Structural HTML
    this.container.innerHTML = `
      <div id="${id}-toggle" class="tab-container" style="margin-bottom: 12px;"></div>

      <div id="${id}-tab-standard" class="tab-content active">
        <div class="select-wrapper model-select-wrapper">
          <select id="${id}-standard-select" class="model-select">${
            RICH_OPTIONS
              ? '<button type="button" class="model-select-button"><selectedcontent></selectedcontent></button>'
              : ''
          }</select>
          <div id="${id}-badge-container" class="model-badge-container" hidden style="position: absolute; top: -9px; right: 10px; display: flex; align-items: center; z-index: 10;">
            <span id="${id}-model-badge" class="model-badge" style="position: static; box-shadow: none; pointer-events: auto; padding-left: 2px;">
              <span class="material-icons model-badge-icon" aria-hidden="true">check_circle</span>
              <button type="button" id="${id}-model-delete-btn" class="model-delete-btn material-icons" title="Delete from cache" style="background: transparent; border: none; padding: 0; margin-right: 2px; cursor: pointer; color: inherit; font-size: 0.9rem; line-height: 1; display: none;">delete_outline</button>
              <span class="model-badge-text">Loaded</span>
            </span>
          </div>
        </div>
        ${
          explicitLoad
            ? `<button type="button" id="${id}-standard-load-btn" class="action-button secondary load-model-btn">Initialize Task</button>
               <div id="${id}-standard-status" class="status-text standard-status"></div>`
            : ''
        }
      </div>

      <div id="${id}-tab-upload" class="tab-content">
        <label class="file-upload-btn">
          <span class="file-upload-label">${this.config.uploadLabel}</span>
          <input type="file" id="${id}-file-input" class="model-upload" accept="${this.config.accept}">
        </label>
        <div id="${id}-upload-status" class="status-text upload-status">No file chosen</div>
      </div>

      <div id="${id}-progress" class="model-loading-progress" style="display: none;">
        <div class="progress-container">
          <div class="progress-bar"></div>
        </div>
        <div class="progress-text">Loading Model... 0%</div>
      </div>
    `;

    // 2. DOM lookups
    const standardTab = this.container.querySelector<HTMLElement>(`#${id}-tab-standard`)!;
    const uploadTab = this.container.querySelector<HTMLElement>(`#${id}-tab-upload`)!;
    this.modelSelect = this.container.querySelector('.model-select')!;
    this.badgeContainer = this.container.querySelector('.model-badge-container')!;
    this.badge = this.container.querySelector('.model-badge')!;
    this.deleteBtn = this.container.querySelector('.model-delete-btn')!;
    this.loadButton = this.container.querySelector('.load-model-btn');
    this.standardStatus = this.container.querySelector('.standard-status');
    this.modelUpload = this.container.querySelector('.model-upload')!;
    this.uploadStatus = this.container.querySelector('.upload-status')!;
    this.progressWrap = this.container.querySelector('.model-loading-progress')!;
    this.progressBar = this.container.querySelector('.progress-bar')!;
    this.progressText = this.container.querySelector('.progress-text')!;
    this.renderOptions();

    // 3. View Toggle Initialization
    new ViewToggle(
      `${id}-toggle`,
      [
        { label: 'Standard', value: 'standard', icon: 'grid_view' },
        { label: 'Upload', value: 'upload', icon: 'upload' },
      ],
      'standard',
      (mode) => {
        this.currentMode = mode as 'standard' | 'upload';
        standardTab.classList.toggle('active', this.currentMode === 'standard');
        uploadTab.classList.toggle('active', this.currentMode === 'upload');
      },
      'tabs'
    );

    // 4. Event Listeners
    this.modelSelect.addEventListener('change', () => {
      this.selectionFixed = true;
      this.modelUpload.value = ''; // clear any uploaded file
      this.uploadStatus.innerText = 'No file chosen';
      if (this.config.autoLoad) {
        this.emit({ type: 'standard', value: this.modelSelect.value });
      } else {
        // A different model was picked: show its badge right away, then load it
        // if it is already on disk or fall back to the button + cache hint.
        this.statusPinned = false;
        this.autoLoadPending = true;
        this.renderModelMarks();
        this.refreshCacheState();
      }
    });

    this.loadButton?.addEventListener('click', () => {
      this.modelUpload.value = '';
      this.uploadStatus.innerText = 'No file chosen';
      this.emit({ type: 'standard', value: this.modelSelect.value });
    });

    this.deleteBtn.addEventListener('click', async () => {
      const selected = this.modelSelect.value;
      if (selected) {
        // Prevent multiple clicks while deleting
        this.deleteBtn.disabled = true;
        try {
          const url = this.config.resolveUrl ? this.config.resolveUrl(selected) : selected;
          if (url) await removeCachedModel(url);
          await this.refreshCacheState();
        } finally {
          this.deleteBtn.disabled = false;
        }
      }
    });

    this.modelUpload.addEventListener('change', (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        this.uploadStatus.innerText = file.name;
        this.emit({ type: 'custom', file });
      }
    });

    this.refreshCacheState();
  }

  private renderOptions() {
    const selected = this.modelSelect.value;
    // Replace only the options (the rich control keeps its <button> child).
    Array.from(this.modelSelect.options).forEach((o) => o.remove());
    this.options.forEach((opt) => {
      const o = document.createElement('option');
      o.value = opt.value;
      o.disabled = Boolean(opt.disabled);
      if (opt.isDefault) o.selected = true;
      if (RICH_OPTIONS) {
        o.innerHTML = `
          <span class="model-label">${opt.label}</span>
          <span class="model-option-actions"></span>
        `;
      } else {
        o.textContent = opt.label;
      }
      this.modelSelect.appendChild(o);
    });
    // Keep the user's pick across option refreshes when it is still available.
    if (selected && this.options.some((opt) => opt.value === selected && !opt.isDefault)) {
      this.modelSelect.value = selected;
    }
    this.renderModelMarks();
  }

  /** State of a standard model: in memory, on disk, or neither. */
  private stateOf(value: string): 'loaded' | 'downloaded' | null {
    if (value && value === this.loadedValue) return 'loaded';
    if (this.cachedValues.has(value)) return 'downloaded';
    return null;
  }

  /**
   * Shows the state of every model: a badge on the dropdown control for the
   * selected one, and a text tag in the option list for all of them (options
   * cannot contain markup).
   */
  private renderModelMarks() {
    for (const option of Array.from(this.modelSelect.options)) {
      const label =
        this.options.find((opt) => opt.value === option.value)?.label ??
        (RICH_OPTIONS ? option.querySelector('.model-label')?.textContent : option.textContent) ??
        '';
      const state = this.stateOf(option.value);
      if (state) {
        option.dataset.state = state;
        option.dataset.tag = state === 'loaded' ? 'Active' : 'Loaded'; // keep for legacy css just in case
      } else {
        delete option.dataset.state;
        delete option.dataset.tag;
      }

      if (RICH_OPTIONS) {
        const actions = option.querySelector('.model-option-actions');
        if (actions) {
          actions.innerHTML = ''; // clear previous
          if (state) {
            const pill = document.createElement('span');
            pill.className = `model-badge-pill ${state === 'loaded' ? 'loaded' : 'downloaded'}`;
            pill.textContent = state === 'loaded' ? 'Active' : 'Loaded';
            actions.appendChild(pill);

            if (state === 'loaded') {
              const checkIcon = document.createElement('span');
              checkIcon.className = 'model-option-icon active-icon material-icons';
              checkIcon.textContent = 'check';
              actions.appendChild(checkIcon);
            } else if (state === 'downloaded') {
              const delBtn = document.createElement('span');
              delBtn.className = 'model-option-icon model-option-delete material-icons';
              delBtn.textContent = 'delete_outline';
              delBtn.title = 'Delete from cache';

              const stopEvent = (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
              };
              delBtn.addEventListener('mousedown', stopEvent);
              delBtn.addEventListener('pointerdown', stopEvent);
              delBtn.addEventListener('mouseup', stopEvent);
              delBtn.addEventListener('pointerup', stopEvent);

              delBtn.addEventListener('click', async (e) => {
                stopEvent(e);
                delBtn.style.pointerEvents = 'none';
                delBtn.style.opacity = '0.5';
                const url = this.config.resolveUrl ? this.config.resolveUrl(option.value) : option.value;
                if (url) await removeCachedModel(url);
                await this.refreshCacheState();
              });
              actions.appendChild(delBtn);
            }
          }
        }
      } else {
        const fallbackTag = state === 'loaded' ? LOADED_TAG : DOWNLOADED_TAG;
        option.textContent = state ? `${label}  ${fallbackTag}` : label;
      }
    }

    const state = this.stateOf(this.modelSelect.value);
    // Only show the floating badge when the selected option is the actively loaded model.
    this.badgeContainer.style.display = state === 'loaded' ? 'flex' : 'none';
    this.updateLoadButton();
    if (!state) return;

    // The badge is now exclusively for the 'loaded' (Active) state.
    const iconEl = this.badge.querySelector('.model-badge-icon') as HTMLElement;
    if (state === 'loaded') {
      iconEl.style.display = 'inline-block';
      this.deleteBtn.style.display = 'none';
      this.badge.querySelector('.model-badge-text')!.textContent = 'Active';
      this.badge.dataset.state = state;
      this.badge.title = 'This model is active and ready to use';
    }
  }

  /**
   * The "Initialize Task" button is only needed when the selected model still has
   * to be fetched (or a previous load failed); while it is the loaded model –
   * or a load is running – there is nothing for it to do.
   */
  private updateLoadButton() {
    if (!this.loadButton) return;
    const loaded = this.stateOf(this.modelSelect.value) === 'loaded';
    this.loadButton.hidden = loaded && !this.busy;
  }

  private emit(selection: ModelSelection) {
    this.lastSelection = selection;
    this.selectionFixed = true;
    // Whatever was loaded before is replaced by this load; the task reports
    // the new model via setLoaded() once it is ready.
    this.loadedValue = null;
    this.renderModelMarks();
    if (selection.type === 'custom') {
      // The upload supersedes the standard status; fall back to the cache hint there.
      this.statusPinned = false;
      if (this.standardStatus) this.standardStatus.textContent = '';
      this.refreshCacheState();
    }
    this.onModelChanged(selection);
  }

  private renderCacheState(url: string | undefined, cached: boolean) {
    const status = this.standardStatus!;
    status.innerHTML = '';
    if (!url) return;
    if (!cached) {
      status.textContent = 'Not loaded yet';
      return;
    }
    status.textContent = 'Loaded · cached in this browser';
  }
}
