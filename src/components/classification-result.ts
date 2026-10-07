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

export interface ClassificationItem {
  label: string;
  score: number; // 0.0 to 1.0
}

export class ClassificationResult {
  private container: HTMLElement;
  private maxRowsRendered = 0;

  constructor(containerId: string) {
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`ClassificationResult: container ${containerId} not found`);
    this.container = el;
    this.injectStyles();
  }

  private injectStyles() {
    if (!document.getElementById('classification-result-styles')) {
      const style = document.createElement('style');
      style.id = 'classification-result-styles';
      style.textContent = `
        .classification-item {
          display: flex;
          align-items: center;
          margin-bottom: 6px;
          padding: 8px 14px;
          background: var(--surface, #fff);
          border-radius: 8px;
          border: 1px solid var(--border-color, #eee);
          box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        .classification-item:last-child {
          margin-bottom: 0;
        }
        .class-name {
          width: 160px;
          flex-shrink: 0;
          font-weight: 600;
          font-size: 13px;
          color: var(--text-main, #333);
          text-transform: capitalize;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .class-bar-container {
          flex-grow: 1;
          background: #f0f2f5;
          height: 8px;
          border-radius: 4px;
          overflow: hidden;
          margin: 0 12px;
        }
        .class-bar {
          height: 100%;
          background: var(--primary, #007f8b);
          border-radius: 4px;
          transition: width 0.5s cubic-bezier(0.4, 0.0, 0.2, 1);
        }
        .class-score {
          width: 42px;
          text-align: right;
          font-family: 'Roboto Mono', monospace;
          font-size: 13px;
          font-weight: 500;
          color: var(--primary, #007f8b);
        }
      `;
      document.head.appendChild(style);
    }
  }

  public updateResults(results: ClassificationItem[]) {
    if (results.length === 0 && this.maxRowsRendered === 0) {
      results = [{ label: 'No results', score: 0 }];
    } else {
      this.maxRowsRendered = Math.max(this.maxRowsRendered, results.length);
    }
    const totalRows = Math.max(this.maxRowsRendered, results.length);

    // Get current rows
    const currentRows = Array.from(this.container.children) as HTMLElement[];

    // Add missing rows
    for (let i = currentRows.length; i < totalRows; i++) {
      const row = document.createElement('div');
      row.className = 'classification-item';
      row.innerHTML = `
        <span class="class-name"></span>
        <div class="class-bar-container">
          <div class="class-bar" style="width: 0%"></div>
        </div>
        <span class="class-score"></span>
      `;
      this.container.appendChild(row);
      currentRows.push(row);
    }

    // Remove extra rows
    while (currentRows.length > totalRows) {
      const row = currentRows.pop();
      row?.remove();
    }

    // Update data
    for (let i = 0; i < totalRows; i++) {
      const row = currentRows[i];
      const nameEl = row.querySelector('.class-name') as HTMLElement;
      const barEl = row.querySelector('.class-bar') as HTMLElement;
      const scoreEl = row.querySelector('.class-score') as HTMLElement;

      if (i < results.length) {
        const result = results[i];
        const scorePercent = Math.round(result.score * 100);
        nameEl.textContent = result.label || 'Unknown';
        barEl.style.width = `${scorePercent}%`;
        barEl.style.background = 'var(--primary, #007f8b)';
        scoreEl.textContent = `${scorePercent}%`;
        scoreEl.style.color = 'var(--primary, #007f8b)';
        nameEl.style.color = 'var(--text-main, #333)';
      } else {
        // Placeholder
        nameEl.textContent = '--';
        barEl.style.width = `0%`;
        barEl.style.background = 'transparent';
        scoreEl.textContent = `--`;
        scoreEl.style.color = 'var(--text-secondary, #666)';
        nameEl.style.color = 'var(--text-secondary, #666)';
      }
    }
  }

  public clear() {
    this.maxRowsRendered = 0;
    this.container.innerHTML = '';
  }
}
