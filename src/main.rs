use coqui_stt::Model;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, StreamConfig, StreamError};
use dasp_interpolate::linear::Linear;
use dasp_signal::interpolate::Converter;
use dasp_signal::Signal;
use ringbuf::{Consumer, HeapRb, SharedRb};
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let host = cpal::default_host();
    let input_device = host.default_input_device().unwrap();
    let input_config: StreamConfig = input_device
        .default_input_config()
        .expect("No supported input configs")
        .into();

    let mut m = Model::new("../../libs/model/model.tflite").unwrap();

    m.enable_external_scorer("../../libs/model/model.scorer")
        .unwrap();
    m.add_hot_word("microwave", 1.0).unwrap();
    m.add_hot_word("toaster", 100.0).unwrap();

    // The buffer to share samples
    let ring = HeapRb::<f32>::new(m.get_sample_rate() as usize * 2);
    let (mut producer, consumer) = ring.split();

    let data_in = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        for &sample in data {
            let _ = producer.push(sample);
        }
    };

    let input_stream =
        match input_device.build_input_stream(&input_config, data_in, handle_error, None) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("Error while starting input stream: {}", err);
                panic!();
            }
        };
    if let Err(err) = input_stream.play() {
        eprintln!("Error while playing input stream: {}", err);
        panic!();
    };

    let conv = SignalWrap { consumer };

    let src_sample_rate = input_config.sample_rate.0 as f64;
    let dest_sample_rate = m.get_sample_rate() as f64;

    // We need to interpolate to the target sample rate
    let interpolator = Linear::new(0, 0);
    let conv = Converter::from_hz_to_hz(conv, interpolator, src_sample_rate, dest_sample_rate);
    let mut conv = conv.until_exhausted();

    let mut audio_buffer = Vec::new();

    let mut sent_talk = false;

    let mut silent_start: Option<Instant> = None;

    loop {
        if let Some(value) = conv.next() {
            audio_buffer.push(value);
            if value > 2000 || value < -2000 {
                if !sent_talk {
                    println!("TALKING");
                    sent_talk = true;
                }
                silent_start = None;
            } else {
                let elapsed = match silent_start {
                    Some(value) => value.elapsed().as_secs_f32(),
                    None => {
                        silent_start = Some(Instant::now());
                        continue;
                    }
                };

                if elapsed >= 5.0 {
                    println!("PROCESSING");
                    sent_talk = false;

                    let st = Instant::now();

                    if input_config.channels == 2 {
                        audio_buffer = stereo_to_mono(&audio_buffer);
                    } else if input_config.channels != 1 {
                        panic!("Expected 1 or 2 audio channels")
                    }

                    // Run the speech to text algorithm
                    let result = m.speech_to_text(&audio_buffer).unwrap();

                    let et = Instant::now();
                    let tt = et.duration_since(st);

                    // Output the result
                    println!("{}", result);
                    println!("took {}ms", tt.as_millis());
                    audio_buffer.clear();
                    silent_start = None;
                }
            }
        }
    }
}

fn handle_error(error: StreamError) {
    eprint!("Error while streaming: {}", error);
}

pub struct SignalWrap {
    consumer: Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>,
}

impl Signal for SignalWrap {
    type Frame = i16;

    fn next(&mut self) -> Self::Frame {
        loop {
            let value = match self.consumer.pop() {
                Some(value) => value,
                None => continue,
            };
            return i16::from_sample(value);
        }
    }
}

fn stereo_to_mono(samples: &[i16]) -> Vec<i16> {
    // converting stereo to mono audio is relatively simple
    // just take the average of the two channels
    samples
        .chunks(2)
        .map(|c| {
            if c.len() == 1 {
                return c[0];
            } else if c.len() == 0 {
                return 0;
            }

            (c[0] + c[1]) / 2
        })
        .collect()
}
