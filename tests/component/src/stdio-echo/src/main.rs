use std::io::{self, BufRead, Write};
fn main() {
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line).ok();
    let trimmed = line.trim_end_matches('\n');
    println!("echo: {}", trimmed);
    io::stdout().flush().ok();
}
