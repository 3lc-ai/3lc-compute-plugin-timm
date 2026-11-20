# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Execute the shipped table reader and preset application without a worker/GPU."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = (Path(__file__).parents[1] / "src/tlc_plugin_timm/ui.html").read_text()


def function(name: str) -> str:
    start = UI.index(f"function {name}(")
    end = re.search(r"^}\n", UI[start:], re.MULTILINE)
    assert end
    return UI[start : start + end.end()]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_table_resolution_and_preset_application(tmp_path):
    source = "\n".join(
        function(name)
        for name in [
            "tmDetectTask",
            "tmApplyFormData",
            "tmOnTrainTableChange",
        ]
    )
    harness = r"""
const assert = require('node:assert/strict');
const elements = {};
const document = { getElementById: id => elements[id] ||= {value:'', style:{}, checked:false, textContent:''} };
let fetches = 0, pending = [], handled = false, resolved = [];
const _tmPost = () => { fetches++; return new Promise(r => pending.push(r)); };
let _tmTableRead = 0, _tmDetectDebounce = null, tmModelsCache = {};
let detectedTask = null, trTaskError = '', trDetecting = false;
let tmDetectedTask = null, tmDetectedImageCol = '', tmDetectedLabelCol = '', tmTaskError = '', tmDetecting = false;
const tmPresets = { tableResolved: async (url, info) => { resolved.push({url, info}); return handled; } };
const tmGate = () => {};
const tmShowTask = () => {};
const _tmSetTaskBadge = () => {};
const _tmShowDetected = () => {};
const _tmEnforceTaskMode = () => {};
const tmSetMode = () => {};
const tmCheckTableAgainstRunTarget = () => {};
const tmScheduleAutoSave = () => {};
const trCollectFormData = () => ({params:{}});
const trPopulateModelDropdown = (task, cb, wanted) => {
  document.getElementById('tr-model-select').value = wanted || ''; if(cb) cb();
};
const trLoadModelParams = async () => {};
const tmSelectModel = async name => { document.getElementById('tm-model-search').value = name; };
let restoredParams;
const _tmRestoreParams = params => { restoredParams = params; };
const train = document.getElementById('tm-train-url');
const project = document.getElementById('tm-3lc-project');
const info = name => ({task:'detection', project_name:name, project_url:'s3://bucket/'+name});
"""
    body = r"""
(async () => {
  // A -> B -> A: URL equality alone must not accept the old A response.
  train.value='a'; const a1=tmDetectTask();
  train.value='b'; const b=tmDetectTask();
  train.value='a'; const a2=tmDetectTask();
  pending[2](info('fresh A')); await a2;
  pending[0](info('stale A')); pending[1](info('stale B')); await Promise.all([a1,b]);
  assert.equal(project.value,'fresh A'); assert.equal(resolved.length,1);
  // Typing invalidates a read immediately, before the next debounce fires.
  train.value='old'; const old=tmDetectTask();
  train.value='new'; tmOnTrainTableChange(); clearTimeout(_tmDetectDebounce);
  pending[3](info('old')); await old; assert.equal(resolved.length,1);
  // A reset reuses the successful metadata lookup and clears stale form values.
  document.getElementById('tm-val-url').value='old validation';
  document.getElementById('tm-run-name').value='old run';
  document.getElementById('tm-pretrained-url').value='old.pt';
  const before=fetches;
  await tmApplyFormData({train_table_url:'b',project_name:'B',params:{}}, {url:'b',info:info('B')});
  assert.equal(fetches,before);
  assert.equal(project.value,'B');
  for (const id of ['tm-val-url','tm-run-name','tm-pretrained-url']) assert.equal(document.getElementById(id).value,'');
  // An explicit saved output override survives metadata refresh.
  await tmApplyFormData({train_table_url:'b',project_name:'custom output',params:{}}, {url:'b',info:info('B')});
  assert.equal(project.value,'custom output');
  // If the shared controller applied the reset, the outer reader must not overwrite it.
  handled=true; train.value='c'; const c=tmDetectTask();
  pending[4](info('C')); assert.equal(await c,false);
  assert.equal(project.value,'custom output');
  process.stdout.write(JSON.stringify({ok:true}));
})().catch(e => {console.error(e); process.exitCode=1;});
"""
    script = tmp_path / "table-transitions.js"
    script.write_text(harness + source + body)
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"ok": True}
