use std::collections::HashMap;
use rand::Rng;
use serde_json::json;

mod donor_agent;
use donor_agent::{Action, DonorAgent, DONATE_BENEFIT, DONATE_COST,
                  REPRODUCE_COST, REPRODUCE_THRESHOLD, S2_CONFIDENCE_THRESHOLD};

const OLLAMA_URL: &str = "http://localhost:11434/api/chat";

// ─── World ────────────────────────────────────────────────────────────────────

struct DonorWorld {
    agents: Vec<DonorAgent>,
    tick: u64,
    next_id: u64,
    coop_history: Vec<f32>,
    pop_history: Vec<usize>,
    energy_history: Vec<f32>,
    repro_mode: String,      // "energy" (v7 baseline) | "qd" (v8 diversity-preserving)
    reinforce_mode: String,  // "std" (v7 baseline) | "divprotect" (v8 reinforce-level diversity)
}

impl DonorWorld {
    fn new(agents: Vec<DonorAgent>, repro_mode: String, reinforce_mode: String) -> Self {
        let n = agents.len() as u64;
        DonorWorld { agents, tick: 0, next_id: n,
            coop_history: vec![], pop_history: vec![], energy_history: vec![],
            repro_mode, reinforce_mode }
    }

    fn run_tick(&mut self, mode: &str) {
        self.tick += 1;
        let mut rng = rand::thread_rng();
        let n = self.agents.len();
        if n < 2 { return; }

        let world_coop = self.coop_history.last().copied().unwrap_or(0.5);

        // Sample K = n donor-recipient pairs
        let pairs: Vec<(usize, usize)> = (0..n)
            .map(|_| {
                let d = rng.gen_range(0..n);
                let mut r = rng.gen_range(0..n);
                while r == d { r = rng.gen_range(0..n); }
                (d, r)
            })
            .collect();

        // Decide actions for all pairs; also capture recipient rep for Hebbian update
        let mut actions: Vec<Action> = Vec::with_capacity(pairs.len());
        let mut recipient_reps: Vec<f32> = Vec::with_capacity(pairs.len());
        for &(di, ri) in &pairs {
            let recipient_id = self.agents[ri].id.clone();
            let rep_snapshot = self.agents[di].reputation.get(&recipient_id).copied().unwrap_or(0.0);
            recipient_reps.push(rep_snapshot);
            let action = match mode {
                "single_llm" => {
                    let ctx = format!("energy={:.0}, no memory, world_coop={:.0}%",
                        self.agents[di].energy, world_coop * 100.0);
                    let a = llm_decide(&self.agents[di].gene, &ctx);
                    self.agents[di].s2_calls += 1;
                    a
                }
                "llm_society" => {
                    let rep = self.agents[di].reputation.get(&recipient_id).copied().unwrap_or(0.0);
                    let ctx = format!("energy={:.0}, recipient_rep={:+.2}, world_coop={:.0}%",
                        self.agents[di].energy, rep, world_coop * 100.0);
                    let a = llm_decide(&self.agents[di].gene, &ctx);
                    self.agents[di].s2_calls += 1;
                    a
                }
                _ => { // atom_hybrid
                    let (s1_action, conf) = self.agents[di].s1_decide(&recipient_id);
                    if conf >= S2_CONFIDENCE_THRESHOLD || self.agents[di].s2_cooldown > 0 {
                        self.agents[di].s1_calls += 1;
                        s1_action
                    } else {
                        let rep = self.agents[di].reputation.get(&recipient_id).copied().unwrap_or(0.0);
                        let ctx = format!("energy={:.0}, recipient_rep={:+.2}, world_coop={:.0}%",
                            self.agents[di].energy, rep, world_coop * 100.0);
                        let a = llm_decide(&self.agents[di].gene, &ctx);
                        self.agents[di].s2_calls += 1;
                        self.agents[di].s2_cooldown = 5;
                        a
                    }
                }
            };
            actions.push(action);
        }

        let mut donate_count = 0usize;

        for (&(di, ri), action) in pairs.iter().zip(actions.iter()) {
            match action {
                Action::Donate => {
                    self.agents[di].energy -= DONATE_COST;
                    self.agents[ri].energy += DONATE_BENEFIT;
                    self.agents[di].donated += 1;
                    self.agents[ri].received_donations += 1;
                    donate_count += 1;
                    let did = self.agents[di].id.clone();
                    // ri directly observes di as cooperator
                    self.agents[ri].observe_action(&did, true);
                    // Gossip: broadcast di's donation (di is the cooperator)
                    if rng.r#gen::<f32>() < 0.8 {
                        let did2 = did.clone();
                        for (idx, agent) in self.agents.iter_mut().enumerate() {
                            if idx != ri {
                                agent.observe_action(&did2, true);
                            }
                        }
                    }
                }
                Action::Defect => {
                    self.agents[di].defected += 1;
                    let did = self.agents[di].id.clone();
                    // Gossip: broadcast di's defection (di is the defector)
                    if rng.r#gen::<f32>() < 0.8 {
                        let did2 = did.clone();
                        for (idx, agent) in self.agents.iter_mut().enumerate() {
                            if idx != di { agent.observe_action(&did2, false); }
                        }
                    }
                }
            }
        }

        // Hebbian reinforcement (atom_hybrid only — S1 learns social correctness)
        if mode == "atom_hybrid" {
            let div_protect = self.reinforce_mode == "divprotect";
            for ((&(di, _), action), &rep) in pairs.iter().zip(actions.iter()).zip(recipient_reps.iter()) {
                self.agents[di].reinforce(action, rep, div_protect);
            }
        }

        // Age + cooldown + survival cost (creates pressure to be selective about donating)
        for a in &mut self.agents {
            a.age += 1;
            a.energy -= 1.0;
            if a.s2_cooldown > 0 { a.s2_cooldown -= 1; }
        }

        // Reproduce — parent selection depends on repro_mode.
        let parents = if self.repro_mode == "qd" {
            select_parents_qd(&self.agents)
        } else {
            select_parents_energy(&self.agents)
        };
        let mut children: Vec<DonorAgent> = vec![];
        for &i in &parents {
            self.agents[i].energy -= REPRODUCE_COST;
            let cid = format!("a{}", self.next_id);
            self.next_id += 1;
            let child = DonorAgent::breed(&self.agents[i], cid);
            children.push(child);
        }
        self.agents.extend(children);

        // Die
        self.agents.retain(|a| a.energy > 0.0);

        // Record metrics
        let coop_rate = if !pairs.is_empty() { donate_count as f32 / pairs.len() as f32 } else { 0.0 };
        let mean_e = if !self.agents.is_empty() {
            self.agents.iter().map(|a| a.energy).sum::<f32>() / self.agents.len() as f32
        } else { 0.0 };
        self.coop_history.push(coop_rate);
        self.pop_history.push(self.agents.len());
        self.energy_history.push(mean_e);
    }

    fn metrics(&self) -> Metrics {
        let n = self.coop_history.len();
        let last_20 = if n >= 20 { &self.coop_history[n-20..] } else { &self.coop_history };
        let avg_coop = last_20.iter().sum::<f32>() / last_20.len() as f32;
        let last_e = self.energy_history.last().copied().unwrap_or(0.0);
        let last_pop = self.pop_history.last().copied().unwrap_or(0);
        let max_gen = self.agents.iter().map(|a| a.generation).max().unwrap_or(0);

        let total_s1: u32 = self.agents.iter().map(|a| a.s1_calls).sum();
        let total_s2: u32 = self.agents.iter().map(|a| a.s2_calls).sum();

        let mut gene_counts: HashMap<String, usize> = HashMap::new();
        for a in &self.agents {
            *gene_counts.entry(a.gene.clone()).or_default() += 1;
        }

        // Strategy-weight diversity (monoculture check) over surviving population.
        let pop = self.agents.len();
        let (mut mean_w, mut std_w, mut wdiv) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
        if pop > 0 {
            for a in &self.agents {
                for k in 0..3 { mean_w[k] += a.strategy_weights[k]; }
            }
            for k in 0..3 { mean_w[k] /= pop as f32; }
            for a in &self.agents {
                for k in 0..3 {
                    let d = a.strategy_weights[k] - mean_w[k];
                    std_w[k] += d * d;
                }
            }
            for k in 0..3 { std_w[k] = (std_w[k] / pop as f32).sqrt(); }
            // mean pairwise Euclidean distance: monoculture → ~0, diverse → larger
            if pop > 1 {
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                for i in 0..pop {
                    for j in (i + 1)..pop {
                        let mut d2 = 0.0f32;
                        for k in 0..3 {
                            let d = self.agents[i].strategy_weights[k] - self.agents[j].strategy_weights[k];
                            d2 += d * d;
                        }
                        sum += d2.sqrt();
                        cnt += 1;
                    }
                }
                wdiv = sum / cnt as f32;
            }
        }

        Metrics { avg_coop_last20: avg_coop, mean_energy: last_e,
            population: last_pop, max_generation: max_gen,
            s1_calls: total_s1, s2_calls: total_s2, gene_counts,
            mean_weights: mean_w, weight_std: std_w, weight_diversity: wdiv }
    }
}

struct Metrics {
    avg_coop_last20: f32,
    mean_energy: f32,
    population: usize,
    max_generation: u32,
    s1_calls: u32,
    s2_calls: u32,
    gene_counts: HashMap<String, usize>,
    mean_weights: [f32; 3],
    weight_std: [f32; 3],
    weight_diversity: f32,
}

// ─── LLM Call ─────────────────────────────────────────────────────────────────

fn llm_decide(model: &str, context: &str) -> Action {
    let body = json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": format!(
                "You are an agent in a survival game. You die if energy reaches 0.\n\
                Situation: {}\n\
                Donating costs you 4 energy but gives the recipient 5 energy.\n\
                Each tick you also lose 1 energy to survive.\n\
                Reply with exactly one word: DONATE or DEFECT",
                context
            )
        }],
        "stream": false,
        "think": false,
        "options": {"num_predict": 6}
    });

    if let Ok(resp) = ureq::post(OLLAMA_URL)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
    {
        if let Ok(json) = resp.into_json::<serde_json::Value>() {
            if let Some(content) = json["message"]["content"].as_str() {
                if content.to_uppercase().contains("DONATE") {
                    return Action::Donate;
                }
            }
        }
    }
    Action::Defect
}

// ─── Runner ───────────────────────────────────────────────────────────────────

// ─── Parent selection (v8) ──────────────────────────────────────────────────
// Strategy niche by argmax(strategy_weights): 0=reciprocate, 1=generous, 2=defect.
fn niche(w: &[f32; 3]) -> usize {
    let mut m = 0;
    for k in 1..3 { if w[k] > w[m] { m = k; } }
    m
}

// v7 baseline: every energy-eligible agent reproduces. This is the mechanism that
// produced the generous-monoculture + robustness cliff at ~1/3..0.40 defector init.
fn select_parents_energy(agents: &[DonorAgent]) -> Vec<usize> {
    (0..agents.len())
        .filter(|&i| agents[i].energy >= REPRODUCE_THRESHOLD)
        .collect()
}

// v8 diversity-preserving selection (QD / coop-Elo).
//
// HYPOTHESIS UNDER TEST: keeping reciprocators (niche 0) alive — instead of letting
// the "generous, always-donate" niche (1) monopolize reproduction — extends the
// robustness frontier past 0.40 defector invasion.
//
// CONSTRAINTS the returned indices must respect:
//   - Only energy-eligible agents (energy >= REPRODUCE_THRESHOLD) may be returned;
//     each pays REPRODUCE_COST, so non-eligible agents would go negative and die.
//   - Return a Vec<usize> of agent indices that will each spawn one child.
//   - breed() clones the parent's niche (+noise), so WHICH niche reproduces decides
//     the next generation's strategy mix. That is the lever.
//
// Helpers available: niche(&w) -> 0|1|2, agents[i].strategy_weights, .energy,
//   .donated / .defected / .received_donations, .reputation.
//
// >>> boo writes this (5-10 lines). See trade-off menu in chat. <<<
fn select_parents_qd(agents: &[DonorAgent]) -> Vec<usize> {
    let n = agents.len();
    let eligible: Vec<usize> = (0..n)
        .filter(|&i| agents[i].energy >= REPRODUCE_THRESHOLD)
        .collect();
    // Split eligible by niche: generous (1) vs the rest (reciprocate 0 + defect 2).
    let (mut generous, mut others): (Vec<usize>, Vec<usize>) = eligible
        .into_iter()
        .partition(|&i| niche(&agents[i].strategy_weights) == 1);
    // Brake generous ONLY when it already dominates the living population (>60%),
    // i.e. when the monoculture is forming. Healthy/early populations behave exactly
    // like baseline → no population-size confound in the robustness measurement.
    // Reciprocators + defectors always reproduce (protect the minority cooperative niche).
    let gen_living = (0..n).filter(|&i| niche(&agents[i].strategy_weights) == 1).count();
    if n > 0 && gen_living as f32 / n as f32 > 0.6 {
        generous.sort_by(|&a, &b| agents[b].energy.partial_cmp(&agents[a].energy).unwrap());
        generous.truncate(others.len().max(2)); // trickle floor so reproduction never fully stalls
    }
    others.extend(generous);
    others
}

fn run_condition(mode: &str, ticks: u32, n_agents: u32, genes: &[&str], def_frac: f32, repro_mode: &str, reinforce_mode: &str) -> Metrics {
    // First n_def agents start with defector-leaning weights (invasion seed).
    // def_frac is the initial defector fraction; baseline = 0.2 (was hardcoded i%5==4).
    let n_def = (n_agents as f32 * def_frac).round() as u32;
    let agents: Vec<DonorAgent> = (0..n_agents)
        .map(|i| {
            let gene = genes[i as usize % genes.len()].to_string();
            let mut agent = DonorAgent::new(format!("a{}", i), gene);
            if i < n_def {
                // defector: high defect weight → S1 confidence > threshold → always uses S1
                agent.strategy_weights = [0.05, 0.05, 0.9];
            }
            agent
        })
        .collect();

    let mut world = DonorWorld::new(agents, repro_mode.to_string(), reinforce_mode.to_string());

    let progress_interval = ticks / 5;
    for t in 0..ticks {
        world.run_tick(mode);
        if (t + 1) % progress_interval == 0 {
            let coop = world.coop_history.last().copied().unwrap_or(0.0);
            let pop = world.pop_history.last().copied().unwrap_or(0);
            println!("  tick {:>4} | coop {:.0}% | pop {:>3}", t + 1, coop * 100.0, pop);
        }
    }

    world.metrics()
}

fn print_metrics(_mode: &str, m: &Metrics) {
    println!("\n  avg_coop (last 20t) : {:.1}%", m.avg_coop_last20 * 100.0);
    println!("  mean_energy         : {:.1}", m.mean_energy);
    println!("  population          : {}", m.population);
    println!("  max_generation      : G{}", m.max_generation);
    if m.s1_calls + m.s2_calls > 0 {
        let total = m.s1_calls + m.s2_calls;
        println!("  S1 / S2 ratio       : {}/{} ({:.0}% S1)",
            m.s1_calls, m.s2_calls, m.s1_calls as f32 / total as f32 * 100.0);
    }
    if !m.gene_counts.is_empty() {
        let mut genes: Vec<_> = m.gene_counts.iter().collect();
        genes.sort_by_key(|(g, _)| g.to_string());
        let gene_str: Vec<_> = genes.iter().map(|(g, c)| {
            let short = g.split(':').last().unwrap_or(g);
            format!("{} ×{}", short, c)
        }).collect();
        println!("  gene distribution   : {}", gene_str.join(", "));
    }
    println!("  weight diversity    : {:.4} (mean pairwise dist; ~0 = monoculture)", m.weight_diversity);
    println!("  mean weights [r,g,d]: [{:.3}, {:.3}, {:.3}]",
        m.mean_weights[0], m.mean_weights[1], m.mean_weights[2]);
}

fn print_sweep_line(def_frac: f32, rep: u32, repro: &str, rein: &str, m: &Metrics) {
    // Machine-parseable line for the invasion-robustness sweep.
    println!(
        "SWEEP repro={} rein={} frac={:.2} rep={} coop={:.1} pop={} gen={} s2={} wdiv={:.4} wmean=[{:.3},{:.3},{:.3}] wstd=[{:.3},{:.3},{:.3}]",
        repro, rein, def_frac, rep, m.avg_coop_last20 * 100.0, m.population, m.max_generation, m.s2_calls,
        m.weight_diversity,
        m.mean_weights[0], m.mean_weights[1], m.mean_weights[2],
        m.weight_std[0], m.weight_std[1], m.weight_std[2],
    );
}

fn main() {
    let ticks: u32 = std::env::args()
        .nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let n_agents: u32 = std::env::args()
        .nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    // arg3: initial defector fraction (default 0.2 = baseline). arg4: condition ("all" | "c").
    let def_frac: f32 = std::env::args()
        .nth(3).and_then(|s| s.parse().ok()).unwrap_or(0.2);
    let cond: String = std::env::args().nth(4).unwrap_or_else(|| "all".into());
    let rep: u32 = std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    // arg6: reproduction mode — "energy" (v7 baseline) | "qd" (v8 diversity-preserving).
    let repro: String = std::env::args().nth(6).unwrap_or_else(|| "energy".into());
    // arg7: reinforce mode — "std" (v7 baseline) | "divprotect" (v8 reinforce-level diversity).
    let rein: String = std::env::args().nth(7).unwrap_or_else(|| "std".into());

    let genes = ["qwen3:0.6b", "qwen3:8b", "llama3.2:3b"];

    println!("=== Donor Game World ===");
    println!("ticks={} agents={} def_frac={:.2} cond={} repro={} rein={} genes={}",
        ticks, n_agents, def_frac, cond, repro, rein, genes.join(", "));
    println!("cost={} benefit={} reproduce@{}\n",
        DONATE_COST, DONATE_BENEFIT, REPRODUCE_THRESHOLD);

    // Sweep mode: condition C only (S1-heavy, lazy LLM → fast), for invasion robustness.
    if cond == "c" {
        println!("─── Condition C: Atom Hybrid (S1 + S2 + evolution) ───");
        let mc = run_condition("atom_hybrid", ticks, n_agents, &genes, def_frac, &repro, &rein);
        print_metrics("atom_hybrid", &mc);
        print_sweep_line(def_frac, rep, &repro, &rein, &mc);
        return;
    }

    // Condition A: Single LLM (no memory, no evolution)
    println!("─── Condition A: Single LLM (no memory, no evolution) ───");
    let ma = run_condition("single_llm", ticks, n_agents, &genes[..1], def_frac, &repro, &rein);
    print_metrics("single_llm", &ma);

    // Condition B: LLM Society (memory + LLM per agent, no Hebbian)
    println!("\n─── Condition B: LLM Society (memory, no Hebbian) ───");
    let mb = run_condition("llm_society", ticks, n_agents, &genes, def_frac, &repro, &rein);
    print_metrics("llm_society", &mb);

    // Condition C: Atom Hybrid (S1 Hebbian + lazy S2 LLM + evolution)
    println!("\n─── Condition C: Atom Hybrid (S1 + S2 + evolution) ───");
    let mc = run_condition("atom_hybrid", ticks, n_agents, &genes, def_frac, &repro, &rein);
    print_metrics("atom_hybrid", &mc);

    // Summary comparison
    println!("\n=== Summary ===");
    println!("{:<20} {:>10} {:>12} {:>8} {:>8}",
        "Condition", "Coop%", "MeanEnergy", "Pop", "MaxGen");
    println!("{}", "─".repeat(60));
    for (name, m) in [("A: Single LLM", &ma), ("B: LLM Society", &mb), ("C: Atom Hybrid", &mc)] {
        println!("{:<20} {:>9.1}% {:>12.1} {:>8} {:>8}",
            name, m.avg_coop_last20 * 100.0, m.mean_energy, m.population, m.max_generation);
    }
}
