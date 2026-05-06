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
}

impl DonorWorld {
    fn new(agents: Vec<DonorAgent>) -> Self {
        let n = agents.len() as u64;
        DonorWorld { agents, tick: 0, next_id: n,
            coop_history: vec![], pop_history: vec![], energy_history: vec![] }
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
                    if rng.gen::<f32>() < 0.8 {
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
                    if rng.gen::<f32>() < 0.8 {
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
            for ((&(di, _), action), &rep) in pairs.iter().zip(actions.iter()).zip(recipient_reps.iter()) {
                self.agents[di].reinforce(action, rep);
            }
        }

        // Age + cooldown + survival cost (creates pressure to be selective about donating)
        for a in &mut self.agents {
            a.age += 1;
            a.energy -= 1.0;
            if a.s2_cooldown > 0 { a.s2_cooldown -= 1; }
        }

        // Reproduce
        let mut children: Vec<DonorAgent> = vec![];
        for agent in &mut self.agents {
            if agent.energy >= REPRODUCE_THRESHOLD {
                agent.energy -= REPRODUCE_COST;
                let cid = format!("a{}", self.next_id);
                self.next_id += 1;
                children.push(DonorAgent::breed(agent, cid));
            }
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

        Metrics { avg_coop_last20: avg_coop, mean_energy: last_e,
            population: last_pop, max_generation: max_gen,
            s1_calls: total_s1, s2_calls: total_s2, gene_counts }
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

fn run_condition(mode: &str, ticks: u32, n_agents: u32, genes: &[&str]) -> Metrics {
    // 1/3 agents start with defector-leaning weights to create initial diversity
    let agents: Vec<DonorAgent> = (0..n_agents)
        .map(|i| {
            let gene = genes[i as usize % genes.len()].to_string();
            let mut agent = DonorAgent::new(format!("a{}", i), gene);
            if i % 5 == 4 {
                // defector minority (1/5): high defect weight → S1 confidence > threshold → always uses S1
                agent.strategy_weights = [0.05, 0.05, 0.9];
            }
            agent
        })
        .collect();

    let mut world = DonorWorld::new(agents);

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
}

fn main() {
    let ticks: u32 = std::env::args()
        .nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let n_agents: u32 = std::env::args()
        .nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    let genes = ["qwen3:0.6b", "qwen3:8b", "llama3.2:3b"];

    println!("=== Donor Game World ===");
    println!("ticks={} agents={} genes={}", ticks, n_agents, genes.join(", "));
    println!("cost={} benefit={} reproduce@{}\n",
        DONATE_COST, DONATE_BENEFIT, REPRODUCE_THRESHOLD);

    // Condition A: Single LLM (no memory, no evolution)
    println!("─── Condition A: Single LLM (no memory, no evolution) ───");
    let ma = run_condition("single_llm", ticks, n_agents, &genes[..1]);
    print_metrics("single_llm", &ma);

    // Condition B: LLM Society (memory + LLM per agent, no Hebbian)
    println!("\n─── Condition B: LLM Society (memory, no Hebbian) ───");
    let mb = run_condition("llm_society", ticks, n_agents, &genes);
    print_metrics("llm_society", &mb);

    // Condition C: Atom Hybrid (S1 Hebbian + lazy S2 LLM + evolution)
    println!("\n─── Condition C: Atom Hybrid (S1 + S2 + evolution) ───");
    let mc = run_condition("atom_hybrid", ticks, n_agents, &genes);
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
