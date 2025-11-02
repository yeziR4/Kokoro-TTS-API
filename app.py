from kokoro import KPipeline
import soundfile as sf
import tempfile
import os
import uuid
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Kokoro pipeline (lang_code='a' for American English)
# You can adjust this based on your needs
pipeline = None


def get_pipeline():
    """Lazy load pipeline to avoid initialization issues"""
    global pipeline
    if pipeline is None:
        logger.info("Initializing Kokoro TTS pipeline...")
        pipeline = KPipeline(lang_code='a')
        logger.info("Pipeline initialized successfully")
    return pipeline


# Available voices mapping
AVAILABLE_VOICES = {
    'af_heart': 'Female - Heart',
    'af_bella': 'Female - Bella',
    'af_sarah': 'Female - Sarah',
    'af_nicole': 'Female - Nicole',
    'af_sky': 'Female - Sky',
    'am_adam': 'Male - Adam',
    'am_michael': 'Male - Michael',
    'bf_emma': 'British Female - Emma',
    'bf_isabella': 'British Female - Isabella',
    'bm_george': 'British Male - George',
    'bm_lewis': 'British Male - Lewis'
}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Kokoro TTS API',
        'version': '1.0.0'
    }), 200


@app.route('/voices', methods=['GET'])
def list_voices():
    """List all available voices"""
    return jsonify({
        'voices': [{
            'id': voice_id,
            'name': voice_name
        } for voice_id, voice_name in AVAILABLE_VOICES.items()]
    }), 200


@app.route('/generate', methods=['POST'])
def generate_audio():
    """
    Generate audio from text

    Expected JSON body:
    {
        "text": "Your text here",
        "voice": "af_heart",  # optional, defaults to af_heart
        "format": "wav"       # optional, defaults to wav
    }
    """
    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        text = data.get('text', '').strip()
        voice = data.get('voice', 'af_heart')
        audio_format = data.get('format', 'wav')

        # Validation
        if not text:
            return jsonify(
                {'error': 'Text field is required and cannot be empty'}), 400

        if voice not in AVAILABLE_VOICES:
            return jsonify({
                'error':
                f'Invalid voice. Available voices: {list(AVAILABLE_VOICES.keys())}'
            }), 400

        if audio_format not in ['wav', 'mp3']:
            return jsonify({'error': 'Format must be wav or mp3'}), 400

        logger.info(
            f"Generating audio: voice={voice}, text_length={len(text)}")

        # Get pipeline
        pipe = get_pipeline()

        # Generate audio
        generator = pipe(text, voice=voice)

        # Collect all audio chunks
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)
            logger.info(f"Generated chunk {i}: {gs}, {ps}")

        # Concatenate all audio chunks
        import numpy as np
        full_audio = np.concatenate(audio_chunks)

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())
        temp_file = os.path.join(temp_dir,
                                 f'kokoro_{unique_id}.{audio_format}')

        # Save audio file
        sf.write(temp_file, full_audio, 24000)

        logger.info(f"Audio generated successfully: {temp_file}")

        # Return the audio file
        return send_file(temp_file,
                         mimetype=f'audio/{audio_format}',
                         as_attachment=True,
                         download_name=f'generated_{unique_id}.{audio_format}')

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to generate audio',
            'details': str(e)
        }), 500


@app.route('/generate-batch', methods=['POST'])
def generate_batch():
    """
    Generate multiple audio files from multiple text segments

    Expected JSON body:
    {
        "segments": [
            {"text": "First segment", "voice": "af_heart"},
            {"text": "Second segment", "voice": "am_adam"}
        ],
        "format": "wav"  # optional
    }

    Returns: ZIP file containing all generated audio files
    """
    try:
        import zipfile
        import io

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        segments = data.get('segments', [])
        audio_format = data.get('format', 'wav')

        if not segments:
            return jsonify({'error': 'Segments array is required'}), 400

        if audio_format not in ['wav', 'mp3']:
            return jsonify({'error': 'Format must be wav or mp3'}), 400

        logger.info(f"Batch generation: {len(segments)} segments")

        # Get pipeline
        pipe = get_pipeline()

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w',
                             zipfile.ZIP_DEFLATED) as zip_file:
            for idx, segment in enumerate(segments):
                text = segment.get('text', '').strip()
                voice = segment.get('voice', 'af_heart')

                if not text:
                    logger.warning(f"Skipping empty segment {idx}")
                    continue

                if voice not in AVAILABLE_VOICES:
                    voice = 'af_heart'  # Default fallback

                # Generate audio
                generator = pipe(text, voice=voice)
                audio_chunks = []

                for i, (gs, ps, audio) in enumerate(generator):
                    audio_chunks.append(audio)

                # Concatenate audio
                import numpy as np
                full_audio = np.concatenate(audio_chunks)

                # Save to temporary file
                temp_file = f'/tmp/segment_{idx}.{audio_format}'
                sf.write(temp_file, full_audio, 24000)

                # Add to ZIP
                zip_file.write(temp_file, f'segment_{idx:03d}.{audio_format}')

                # Clean up temp file
                os.remove(temp_file)

                logger.info(f"Generated segment {idx}/{len(segments)}")

        zip_buffer.seek(0)

        return send_file(zip_buffer,
                         mimetype='application/zip',
                         as_attachment=True,
                         download_name='podcast_audio_batch.zip')

    except Exception as e:
        logger.error(f"Error in batch generation: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to generate batch audio',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
