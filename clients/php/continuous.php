<?php
// Usage:
// IMAGE=input.png PROMPT='full scene prompt' php continuous.php

$api = rtrim(getenv('API_URL') ?: 'http://localhost:8000', '/');
$image = getenv('IMAGE') ?: ($argv[1] ?? null);
$prompt = getenv('PROMPT') ?: implode(' ', array_slice($argv, 2));
$apiKey = getenv('API_KEY') ?: '';

if (!$image || !$prompt) {
    fwrite(STDERR, "Set IMAGE and PROMPT, or pass image and prompt as arguments.\n");
    exit(2);
}

$headers = $apiKey ? ["x-api-key: {$apiKey}"] : [];
$post = [
    'image' => curl_file_create($image),
    'prompt' => $prompt,
    'directions' => getenv('DIRECTIONS') ?: 'left,right,up,down',
    'expand_pixels' => getenv('EXPAND_PIXELS') ?: '256',
    'steps' => getenv('STEPS') ?: '8',
    'max_steps' => getenv('MAX_STEPS') ?: '20',
    'delay_seconds' => getenv('DELAY_SECONDS') ?: '0',
    'randomize_seed' => getenv('RANDOMIZE_SEED') ?: 'true',
    'seed' => getenv('SEED') ?: '42',
];

$ch = curl_init("{$api}/api/v1/jobs/continuous");
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $post,
    CURLOPT_HTTPHEADER => $headers,
    CURLOPT_RETURNTRANSFER => true,
]);
$body = curl_exec($ch);
$status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
if ($body === false || $status >= 400) {
    throw new RuntimeException($body ?: curl_error($ch));
}
$job = json_decode($body, true, flags: JSON_THROW_ON_ERROR);
echo "Started job {$job['id']}\n";

while (true) {
    sleep(2);
    $ch = curl_init("{$api}{$job['status_url']}");
    curl_setopt_array($ch, [
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_RETURNTRANSFER => true,
    ]);
    $body = curl_exec($ch);
    $update = json_decode($body, true, flags: JSON_THROW_ON_ERROR);
    printf(
        "[%s] step=%d direction=%s image=%s\n",
        $update['status'],
        $update['current_step'],
        $update['latest_direction'] ?? '-',
        $update['latest_url'] ?? '-'
    );
    if (in_array($update['status'], ['completed', 'stopped', 'failed'], true)) {
        break;
    }
}
