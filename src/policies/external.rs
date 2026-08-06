//! External policy: delegates worker selection to a Python callable
//!
//! The callable is installed process-globally by the PyO3 `Router` before
//! startup (see `set_external_hooks`), because `PolicyConfig` is plain data
//! and policies are constructed from it in several places (factory, registry,
//! service discovery). The callable contract is:
//!
//! `select(workers: list[dict], request_text: str | None, headers: dict | None) -> int | None`
//!
//! where each worker dict carries `url`, `model_id`, `worker_type`, `load`,
//! in the exact order of the worker slice passed to the policy. Any error or
//! `None` from Python falls back to an inner policy; a request is never failed
//! by the external hook.

use super::{LoadBalancingPolicy, PolicyFactory, RequestHeaders, RoundRobinPolicy};
use crate::config::PolicyConfig;
use crate::core::{Worker, WorkerType};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, RwLock};
use tracing::{debug, warn};

/// Hooks installed by the Python launcher before the router starts
#[derive(Debug, Clone)]
pub struct ExternalHooks {
    /// Python callable making the selection decision
    pub select: Arc<Py<PyAny>>,
    /// Policy used when the callable declines or fails
    pub fallback: PolicyConfig,
}

static HOOKS: RwLock<Option<ExternalHooks>> = RwLock::new(None);

/// Install the process-global external hooks
pub fn set_external_hooks(hooks: ExternalHooks) {
    *HOOKS.write().unwrap() = Some(hooks);
}

pub(crate) fn external_hooks() -> Option<ExternalHooks> {
    HOOKS.read().unwrap().clone()
}

/// Delegates selection to an external Python scheduler with a Rust fallback
#[derive(Debug)]
pub struct ExternalPolicy {
    select: Arc<Py<PyAny>>,
    fallback: Arc<dyn LoadBalancingPolicy>,
}

impl ExternalPolicy {
    /// Build from the process-global hooks; None if none were installed
    pub fn from_globals() -> Option<Self> {
        let hooks = external_hooks()?;
        let fallback = match &hooks.fallback {
            // Guard against a self-referential fallback config
            PolicyConfig::External => {
                Arc::new(RoundRobinPolicy::new()) as Arc<dyn LoadBalancingPolicy>
            }
            cfg => PolicyFactory::create_from_config(cfg),
        };
        Some(Self {
            select: hooks.select,
            fallback,
        })
    }

    /// Same as from_globals, but returned as the generic policy interface
    /// the factory and registry store
    pub fn from_globals_as_policy() -> Option<Arc<dyn LoadBalancingPolicy>> {
        let policy = Self::from_globals()?;
        Some(Arc::new(policy))
    }

    fn call_select(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> PyResult<Option<usize>> {
        Python::attach(|py| {
            let worker_dicts = PyList::empty(py);
            for worker in workers {
                let entry = PyDict::new(py);
                entry.set_item("url", worker.url())?;
                entry.set_item("model_id", worker.model_id())?;
                entry.set_item("worker_type", worker_type_str(&worker.worker_type()))?;
                entry.set_item("load", worker.load())?;
                worker_dicts.append(entry)?;
            }

            let header_dict = match headers {
                Some(h) => {
                    let dict = PyDict::new(py);
                    for (key, value) in h {
                        dict.set_item(key, value)?;
                    }
                    Some(dict)
                }
                None => None,
            };

            self.select
                .bind(py)
                .call1((worker_dicts, request_text, header_dict))?
                .extract::<Option<usize>>()
        })
    }
}

fn worker_type_str(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

impl LoadBalancingPolicy for ExternalPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        if workers.is_empty() {
            return None;
        }
        match self.call_select(workers, request_text, headers) {
            Ok(Some(idx)) if idx < workers.len() => Some(idx),
            Ok(Some(idx)) => {
                warn!(
                    "External policy returned out-of-range index {} for {} workers, using fallback",
                    idx,
                    workers.len()
                );
                self.fallback
                    .select_worker_with_headers(workers, request_text, headers)
            }
            Ok(None) => {
                debug!("External policy declined, using fallback");
                self.fallback
                    .select_worker_with_headers(workers, request_text, headers)
            }
            Err(err) => {
                warn!("External policy call failed: {}, using fallback", err);
                self.fallback
                    .select_worker_with_headers(workers, request_text, headers)
            }
        }
    }

    fn name(&self) -> &'static str {
        "external"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
