#include "audio.h"
#include "components.h"

#include <AL/al.h>
#include <AL/alc.h>

#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace {
    constexpr float kPi = 3.14159265358979323846f;

    struct WavData {
        std::vector<unsigned char> pcm;
        int channels = 0;
        int sampleRate = 0;
        int bits = 0;
        bool ok = false;
    };

    uint32_t rd32(std::ifstream& f) {
        unsigned char b[4]; f.read(reinterpret_cast<char*>(b), 4);
        return b[0] | (b[1] << 8) | (b[2] << 16) | (uint32_t(b[3]) << 24);
    }
    uint16_t rd16(std::ifstream& f) {
        unsigned char b[2]; f.read(reinterpret_cast<char*>(b), 2);
        return static_cast<uint16_t>(b[0] | (b[1] << 8));
    }

    // Minimal RIFF/WAVE PCM loader (the only format the engine ships/needs).
    WavData parseWav(const std::string& path) {
        WavData w;
        std::ifstream f(path, std::ios::binary);
        if (!f) return w;
        char tag[4];
        f.read(tag, 4);                       if (std::memcmp(tag, "RIFF", 4)) return w;
        rd32(f);                              // overall size (ignored)
        f.read(tag, 4);                       if (std::memcmp(tag, "WAVE", 4)) return w;

        uint16_t fmt = 0;
        while (f && f.read(tag, 4)) {
            uint32_t sz = rd32(f);
            if (!std::memcmp(tag, "fmt ", 4)) {
                fmt          = rd16(f);       // 1 = PCM
                w.channels   = rd16(f);
                w.sampleRate = static_cast<int>(rd32(f));
                rd32(f);                      // byte rate
                rd16(f);                      // block align
                w.bits       = rd16(f);
                for (uint32_t i = 16; i < sz; ++i) f.get();   // skip extra fmt
            } else if (!std::memcmp(tag, "data", 4)) {
                w.pcm.resize(sz);
                f.read(reinterpret_cast<char*>(w.pcm.data()), sz);
                break;
            } else {
                f.seekg(sz + (sz & 1), std::ios::cur);        // skip + pad
            }
        }
        w.ok = fmt == 1 && !w.pcm.empty() && (w.bits == 8 || w.bits == 16) &&
               (w.channels == 1 || w.channels == 2);
        if (!w.ok)
            std::fprintf(stderr, "[Audio] WAV load failed/unsupported: %s\n",
                         path.c_str());
        return w;
    }

    ALenum alFormat(int channels, int bits) {
        if (channels == 1) return bits == 8 ? AL_FORMAT_MONO8  : AL_FORMAT_MONO16;
        return                     bits == 8 ? AL_FORMAT_STEREO8 : AL_FORMAT_STEREO16;
    }
}

struct AudioEngine::Impl {
    ALCdevice*  device  = nullptr;
    ALCcontext* context = nullptr;
    bool        ok      = false;
    std::unordered_map<std::string, ALuint> wavCache;
    std::vector<ALuint> buffers;
    std::vector<ALuint> sources;
};

AudioEngine::AudioEngine()  = default;
AudioEngine::~AudioEngine() { cleanup(); }

bool AudioEngine::ok() const { return m_impl && m_impl->ok; }

void AudioEngine::init() {
    if (m_impl) return;
    m_impl = std::make_unique<Impl>();

    m_impl->device = alcOpenDevice(nullptr);
    if (!m_impl->device) { std::fprintf(stderr, "[Audio] no OpenAL device\n"); return; }
    m_impl->context = alcCreateContext(m_impl->device, nullptr);
    if (!m_impl->context || !alcMakeContextCurrent(m_impl->context)) {
        std::fprintf(stderr, "[Audio] failed to create/make context\n");
        return;
    }
    alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);
    // Disable Doppler: the listener is a manually-driven editor camera, so
    // velocity-based pitch shifting just sounds like distortion when the user
    // moves/orbits, not a desirable effect.
    alDopplerFactor(0.0f);
    alListenerf(AL_GAIN, m_masterGain);
    m_impl->ok = true;
}

void AudioEngine::cleanup() {
    if (!m_impl) return;
    if (m_impl->ok) {
        for (ALuint s : m_impl->sources) { alSourceStop(s); alDeleteSources(1, &s); }
        for (ALuint b : m_impl->buffers) alDeleteBuffers(1, &b);
    }
    if (m_impl->context) {
        alcMakeContextCurrent(nullptr);
        alcDestroyContext(m_impl->context);
    }
    if (m_impl->device) alcCloseDevice(m_impl->device);
    m_impl.reset();
}

uint32_t AudioEngine::createTone(float freqHz, float seconds, float amplitude) {
    if (!ok()) return 0;
    const int rate = 44100;
    const int n = static_cast<int>(seconds * rate);
    if (n <= 0) return 0;
    std::vector<int16_t> samples(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / rate;
        samples[i] = static_cast<int16_t>(
            amplitude * 32767.0f * std::sin(2.0f * kPi * freqHz * t));
    }
    ALuint buf = 0;
    alGenBuffers(1, &buf);
    alBufferData(buf, AL_FORMAT_MONO16, samples.data(),
                 static_cast<ALsizei>(samples.size() * sizeof(int16_t)), rate);
    m_impl->buffers.push_back(buf);
    return buf;
}

uint32_t AudioEngine::loadWav(const std::string& path) {
    if (!ok()) return 0;
    auto it = m_impl->wavCache.find(path);
    if (it != m_impl->wavCache.end()) return it->second;

    WavData w = parseWav(path);
    if (!w.ok) return 0;

    ALuint buf = 0;
    alGenBuffers(1, &buf);
    alBufferData(buf, alFormat(w.channels, w.bits), w.pcm.data(),
                 static_cast<ALsizei>(w.pcm.size()), w.sampleRate);
    m_impl->buffers.push_back(buf);
    m_impl->wavCache[path] = buf;
    return buf;
}

void AudioEngine::update(entt::registry& reg,
                         const glm::vec3& lp, const glm::vec3& lf,
                         const glm::vec3& lu, const glm::vec3& lv) {
    if (!ok()) return;

    alListener3f(AL_POSITION, lp.x, lp.y, lp.z);
    alListener3f(AL_VELOCITY, lv.x, lv.y, lv.z);
    float ori[6] = { lf.x, lf.y, lf.z, lu.x, lu.y, lu.z };
    alListenerfv(AL_ORIENTATION, ori);
    alListenerf(AL_GAIN, m_masterGain);

    auto view = reg.view<AudioSourceComponent, TransformComponent>();
    for (auto e : view) {
        auto& a  = view.get<AudioSourceComponent>(e);
        auto& tf = view.get<TransformComponent>(e);

        if (a.sourceId == 0) {
            uint32_t buf = a.clip.empty() ? createTone(a.toneHz, a.toneSeconds)
                                          : loadWav(a.clip);
            if (buf == 0 && !a.clip.empty())            // fall back so it isn't silent
                buf = createTone(a.toneHz, a.toneSeconds);
            if (buf == 0) continue;
            ALuint src = 0;
            alGenSources(1, &src);
            alSourcei(src, AL_BUFFER, static_cast<ALint>(buf));
            a.bufferId = buf;
            a.sourceId = src;
            m_impl->sources.push_back(src);
        }

        ALuint src = a.sourceId;
        alSourcef(src, AL_GAIN,  a.gain);
        alSourcef(src, AL_PITCH, a.pitch);
        alSourcei(src, AL_LOOPING, a.loop ? AL_TRUE : AL_FALSE);
        if (a.spatial) {
            alSourcei(src, AL_SOURCE_RELATIVE, AL_FALSE);
            alSource3f(src, AL_POSITION, tf.position.x, tf.position.y, tf.position.z);
            alSourcef(src, AL_REFERENCE_DISTANCE, a.refDistance);
            alSourcef(src, AL_MAX_DISTANCE, a.maxDistance);
            alSourcef(src, AL_ROLLOFF_FACTOR, a.rolloff);
        } else {
            alSourcei(src, AL_SOURCE_RELATIVE, AL_TRUE);
            alSource3f(src, AL_POSITION, 0.0f, 0.0f, 0.0f);
            alSourcef(src, AL_ROLLOFF_FACTOR, 0.0f);
        }

        ALint state = 0;
        alGetSourcei(src, AL_SOURCE_STATE, &state);
        if (a.playing && state != AL_PLAYING)      alSourcePlay(src);
        else if (!a.playing && state == AL_PLAYING) alSourceStop(src);
    }
}

void AudioEngine::removeSource(entt::registry& reg, entt::entity e) {
    if (!ok()) return;
    auto* a = reg.try_get<AudioSourceComponent>(e);
    if (!a || a->sourceId == 0) return;
    ALuint src = a->sourceId;
    alSourceStop(src);
    alDeleteSources(1, &src);
    auto& v = m_impl->sources;
    v.erase(std::remove(v.begin(), v.end(), src), v.end());
    a->sourceId = 0;
}

void AudioEngine::setMasterGain(float g) {
    m_masterGain = g < 0.0f ? 0.0f : g;
    if (ok()) alListenerf(AL_GAIN, m_masterGain);
}
