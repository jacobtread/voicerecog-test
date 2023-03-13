use audrey::Reader;
use coqui_stt::Model;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, StreamConfig, StreamError};
use dasp_interpolate::linear::Linear;
use dasp_signal::interpolate::Converter;
use dasp_signal::{from_iter, Signal};
use ringbuf::{Consumer, HeapRb, SharedRb};
use std::env::args;
use std::fs::File;
use std::mem::MaybeUninit;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::yield_now;
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

    // The buffer to share samples
    let ring = HeapRb::<f32>::new(m.get_sample_rate() as usize * 2);
    let (mut producer, mut consumer) = ring.split();

    // Fill the samples with 0.0 equal to the length of the delay.
    for _ in 0..m.get_sample_rate() as usize {
        // The ring buffer has twice as much space as necessary to add latency here,
        // so this should never fail
        producer.push(0.0).unwrap();
    }

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

    let mut silent_count = 0;

    let mut sent_talk = false;
    let mut sent_silent = false;

    let mut actual_data = 0;

    loop {
        if let Some(value) = conv.next() {
            audio_buffer.push(value);
            if (value > 1000 || value < -1000) {
                actual_data += 1;
                if !sent_talk {
                    println!("TALKING");
                    sent_talk = true;
                }
            } else {
                silent_count += 1;

                if silent_count > 100000 && audio_buffer.len() > 0 && actual_data > 0 {
                    println!("PROCESSING");
                    sent_talk = false;

                    let st = Instant::now();

                    // Run the speech to text algorithm
                    let result = m.speech_to_text(&audio_buffer).unwrap();

                    let et = Instant::now();
                    let tt = et.duration_since(st);

                    // Output the result
                    println!("{}", result);
                    println!("took {}ns", tt.as_nanos());
                    silent_count = 0;
                    audio_buffer.clear();
                    actual_data = 0;
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

// fn main1() {
//     let audio_file_path = args()
//         .nth(1)
//         .expect("Please specify an audio file to run STT on");

//     let mut m = Model::new("../../libs/model/model.tflite").unwrap();

//     m.enable_external_scorer("../../libs/model/model.scorer")
//         .unwrap();

//     let audio_file = File::open(audio_file_path).unwrap();
//     let mut reader = Reader::new(audio_file).unwrap();
//     let desc = reader.description();
//     // input audio must be mono and usually at 16KHz, but this depends on the model
//     let channel_count = desc.channel_count();

//     let src_sample_rate = desc.sample_rate();
//     let dest_sample_rate = m.get_sample_rate() as u32;
//     // Obtain the buffer of samples
//     let mut audio_buf: Vec<_> = if src_sample_rate == dest_sample_rate {
//         reader.samples().map(|s| s.unwrap()).collect()
//     } else {
//         // We need to interpolate to the target sample rate
//         let interpolator = Linear::new([0i16], [0]);
//         let conv = Converter::from_hz_to_hz(
//             from_iter(reader.samples::<i16>().map(|s| [s.unwrap()])),
//             interpolator,
//             src_sample_rate as f64,
//             dest_sample_rate as f64,
//         );
//         conv.until_exhausted().map(|v| v[0]).collect()
//     };
//     // Convert to mono if required
//     if channel_count == 2 {
//         audio_buf = stereo_to_mono(&audio_buf);
//     } else if channel_count != 1 {
//         panic!(
//             "unknown number of channels: got {}, expected 1 or 2",
//             channel_count
//         );
//     }

//     let st = Instant::now();

//     // Run the speech to text algorithm
//     let result = m.speech_to_text(&audio_buf).unwrap();

//     let et = Instant::now();
//     let tt = et.duration_since(st);

//     // Output the result
//     println!("{}", result);
//     println!("took {}ns", tt.as_nanos());
// }

// fn stereo_to_mono(samples: &[i16]) -> Vec<i16> {
//     // converting stereo to mono audio is relatively simple
//     // just take the average of the two channels
//     samples
//         .chunks(2)
//         .map(|c| {
//             if c.len() == 1 {
//                 return c[0];
//             } else if c.len() == 0 {
//                 return 0;
//             }

//             (c[0] + c[1]) / 2
//         })
//         .collect()
// }
