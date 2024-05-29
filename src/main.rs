use llm::{conversation_inference_callback, InferenceFeedback, InferenceResponse, InferenceStats, ModelArchitecture, TokenizerSource};
use rustyline::error::ReadlineError;
use std::{convert::Infallible, io::Write, path::PathBuf};
use rand::thread_rng;

const LOCAL_MODEL_PATH: &str = "C:/Users/rohan/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin";

#[tokio::main]
async fn main() {
    // Hard-coded configuration values
    let model_architecture = ModelArchitecture::Llama;
    let model_path = PathBuf::from(LOCAL_MODEL_PATH.to_string());
    let tokenizer_path: Option<PathBuf> = None;
    let tokenizer_repository: Option<String> = None;

    let tokenizer_source = match (&tokenizer_path, &tokenizer_repository) {
        (Some(_), Some(_)) => {
            panic!("Cannot specify both tokenizer_path and tokenizer_repository");
        }
        (Some(path), None) => TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
        (None, Some(repo)) => TokenizerSource::HuggingFaceRemote(repo.to_owned()),
        (None, None) => TokenizerSource::Embedded,
    };


    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture:?} model from {model_path:?}: {err}")
    });

    let mut session = model.start_session(Default::default());

    let character_name = "### Assistant";
    let user_name = "### Human";
    let persona = "A chat between a human and an assistant.";
    let history = format!("{character_name}: Hello - How may I help you today?");

    let inference_parameters = llm::InferenceParameters::default();

    {
        session
            .feed_prompt(
                model.as_ref(),
                format!("{persona}\n{history}").as_str(),
                &mut Default::default(),
                llm::feed_prompt_callback(|resp| match resp {
                    InferenceResponse::PromptToken(t) | InferenceResponse::InferredToken(t) => {
                        print_token(t);
                        Ok::<InferenceFeedback, Infallible>(InferenceFeedback::Continue)
                    }
                    _ => Ok(InferenceFeedback::Continue),
                }),
            )
            .expect("Failed to ingest initial prompt.");
    }

    let mut rl = rustyline::DefaultEditor::new().expect("Failed to create input reader");
    let mut rng = thread_rng();
    let mut res = InferenceStats::default();

    loop {
        println!();
        let readline = rl.readline(&format!("{user_name}: "));
        match readline {
            Ok(line) => {
                let stats = {
                    session
                        .infer::<Infallible>(
                            model.as_ref(),
                            &mut rng,
                            &llm::InferenceRequest {
                                prompt: format!("{user_name}: {line}\n{character_name}:").as_str().into(),
                                parameters: &inference_parameters,
                                play_back_previous_tokens: false,
                                maximum_token_count: None,
                            },
                            &mut Default::default(),
                            conversation_inference_callback(&format!("{character_name}:"), print_token),
                        )
                        .unwrap_or_else(|e| panic!("{e}"))
                };

                res.feed_prompt_duration = res.feed_prompt_duration.saturating_add(stats.feed_prompt_duration);
                res.prompt_tokens += stats.prompt_tokens;
                res.predict_duration = res.predict_duration.saturating_add(stats.predict_duration);
                res.predict_tokens += stats.predict_tokens;
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                println!("{err}");
            }
        }
    }

    println!("\n\nInference stats:\n{res}");
}

fn print_token(t: String) {
    print!("{t}");
    std::io::stdout().flush().unwrap();
}