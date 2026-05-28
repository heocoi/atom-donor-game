use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

pub const DONATE_COST: f32 = 4.0;
pub const DONATE_BENEFIT: f32 = 5.0;
pub const INITIAL_ENERGY: f32 = 50.0;
pub const REPRODUCE_THRESHOLD: f32 = 80.0;
pub const REPRODUCE_COST: f32 = 30.0;
pub const S2_CONFIDENCE_THRESHOLD: f32 = 0.65;

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Donate,
    Defect,
}

/// S1 Hebbian weights: 3 cooperative strategies.
/// [reciprocate, generous, defect]
/// Removed 'cautious' (world_coop): creates self-reinforcing feedback loop
/// (low coop → cautious shift to defect → lower coop → collapse).
/// S1 should learn from direct reputation signals, not aggregate statistics.
#[derive(Clone, Serialize, Deserialize)]
pub struct DonorAgent {
    pub id: String,
    pub gene: String,             // Ollama model name
    pub energy: f32,
    pub generation: u32,
    pub age: u32,

    // S1: Hebbian strategy weights [reciprocate, generous, defect], sum to 1.0
    pub strategy_weights: [f32; 3],

    // Reputation: positive = this agent donates, negative = defects
    pub reputation: HashMap<String, f32>,

    pub s2_cooldown: u32,

    pub donated: u32,
    pub defected: u32,
    pub received_donations: u32,
    pub s1_calls: u32,
    pub s2_calls: u32,
}

impl DonorAgent {
    pub fn new(id: String, gene: String) -> Self {
        let mut rng = rand::thread_rng();
        let mut w = [
            0.35 + rng.r#gen::<f32>() * 0.1,
            0.45 + rng.r#gen::<f32>() * 0.1,
            0.10 + rng.r#gen::<f32>() * 0.1,
        ];
        normalize_weights(&mut w);
        DonorAgent {
            id, gene, energy: INITIAL_ENERGY,
            generation: 0, age: 0,
            strategy_weights: w,
            reputation: HashMap::new(),
            s2_cooldown: 0,
            donated: 0, defected: 0, received_donations: 0,
            s1_calls: 0, s2_calls: 0,
        }
    }

    pub fn breed(parent: &DonorAgent, child_id: String) -> Self {
        let mut rng = rand::thread_rng();
        let mut w = parent.strategy_weights;
        for wi in &mut w {
            *wi += rng.gen_range(-0.04..0.04);
            *wi = wi.max(0.01);
        }
        normalize_weights(&mut w);
        DonorAgent {
            id: child_id,
            gene: parent.gene.clone(),
            energy: INITIAL_ENERGY,
            generation: parent.generation + 1,
            age: 0,
            strategy_weights: w,
            reputation: HashMap::new(),
            s2_cooldown: 0,
            donated: 0, defected: 0, received_donations: 0,
            s1_calls: 0, s2_calls: 0,
        }
    }

    /// S1 fast decision. Returns (action, confidence 0..1).
    pub fn s1_decide(&self, recipient_id: &str) -> (Action, f32) {
        let rep = self.reputation.get(recipient_id).copied().unwrap_or(0.0);

        // Discriminator: refuse known defectors (tit-for-tat).
        // Gossip propagates this → defectors get socially excluded.
        if rep < -0.3 {
            return (Action::Defect, 0.9);
        }

        // Each strategy's donate score (0..1)
        let scores: [f32; 3] = [
            (rep + 1.0) / 2.0,     // reciprocate: trust known givers
            0.9,                    // generous: always lean donate
            0.05,                   // defect: almost never donate
        ];

        let donate_tendency: f32 = self.strategy_weights.iter()
            .zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum();

        let confidence = (donate_tendency - 0.5).abs() * 2.0;
        let action = if donate_tendency >= 0.5 { Action::Donate } else { Action::Defect };
        (action, confidence)
    }

    /// Update reputation after observing other agent's action.
    pub fn observe_action(&mut self, other_id: &str, donated: bool) {
        let delta = if donated { 1.0_f32 } else { -1.0 };
        let rep = self.reputation.entry(other_id.to_string()).or_insert(0.0);
        *rep = (*rep * 0.9 + delta * 0.1).clamp(-1.0, 1.0);
    }

    /// Reinforce Hebbian weight based on social correctness.
    /// social_correct: did we make the right call given recipient's reputation?
    pub fn reinforce(&mut self, action: &Action, recipient_rep: f32, div_protect: bool) {
        let social_correct: f32 = match action {
            Action::Donate =>  recipient_rep,   // donate to cooperator = good, to defector = bad
            Action::Defect => -recipient_rep,   // refuse defector = good, exploit cooperator = bad
        };
        let reward = if social_correct > 0.1 { 0.015 }
                     else if social_correct < -0.1 { -0.008 }
                     else { 0.002 }; // unknown rep: tiny nudge toward cooperation
        let idx = match action {
            Action::Donate => {
                if div_protect {
                    // Diversity-preserving credit: reward reciprocate (0) when donating to
                    // a KNOWN cooperator (it conditions on reputation), generous (1) only for
                    // unknown/neutral recipients. Breaks the rich-get-richer drift where
                    // generous (higher initial weight) always wins credit → monoculture.
                    if recipient_rep > 0.1 { 0 } else { 1 }
                } else {
                    // v7 baseline: credit whichever cooperative weight is currently larger.
                    if self.strategy_weights[0] > self.strategy_weights[1] { 0 } else { 1 }
                }
            }
            Action::Defect => 2,
        };
        self.strategy_weights[idx] += reward;
        self.strategy_weights[idx] = self.strategy_weights[idx].max(0.01);
        normalize_weights(&mut self.strategy_weights);
    }
}

fn normalize_weights(w: &mut [f32; 3]) {
    let sum: f32 = w.iter().sum();
    if sum > 0.0 { for wi in w.iter_mut() { *wi /= sum; } }
}
