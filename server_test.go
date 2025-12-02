package mockllm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/kagent-dev/mockllm"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// For now, we'll use a simple approach where we create mocks with JSON-compatible structures
// that can be marshaled to/from the SDK types. This allows us to test the basic functionality
// while using the SDK types in the type definitions.

func TestSimpleOpenAIMock(t *testing.T) {
	// Create a simple config - we'll use JSON marshaling to convert to SDK types
	openaiRequest := openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role: "user",
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hello"),
					},
				},
			},
		},
	}

	openaiResponse := openai.ChatCompletion{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1677652288,
		Model:   "gpt-4o-mini",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello! How can I help you today?",
				},
				FinishReason: "stop",
			},
		},
	}

	// Convert to JSON and back to get SDK-compatible structure
	var mock mockllm.OpenAIMock
	mock.Name = "test-response"
	mock.Response = openaiResponse
	mock.Match = mockllm.OpenAIRequestMatch{
		MatchType: mockllm.MatchTypeExact,
		Message:   openaiRequest.Messages[len(openaiRequest.Messages)-1],
	}

	// Marshal and unmarshal the request to get it in the right format
	reqBytes, err := json.Marshal(openaiRequest)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytes, &mock.Match)
	require.NoError(t, err)

	config := mockllm.Config{
		OpenAI: []mockllm.OpenAIMock{mock},
	}

	// Start server
	server := mockllm.NewServer(config)
	baseURL, err := server.Start(t.Context())
	require.NoError(t, err)
	defer server.Stop(context.Background()) //nolint:errcheck

	// Make request
	req, err := http.NewRequest("POST", baseURL+"/v1/chat/completions", bytes.NewReader(reqBytes))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	// Check response
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-123", responseBody["id"])
	assert.Equal(t, "chat.completion", responseBody["object"])
}

func TestSimpleAnthropicMock(t *testing.T) {
	// Create a simple config - we'll use JSON marshaling to convert to SDK types
	anthropicRequest := anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-20240620",
		MaxTokens: 1000,
		Messages: []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfText: &anthropic.TextBlockParam{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	anthropicResponse := anthropic.Message{
		ID:   "msg_123",
		Type: "message",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "text",
				Text: "Hello! How can I assist you today?",
			},
		},
		Model:      "claude-3-5-sonnet-20240620",
		StopReason: "end_turn",
	}

	// Convert to JSON and back to get SDK-compatible structure
	var mock mockllm.AnthropicMock
	mock.Name = "test-response"
	mock.Response = anthropicResponse
	mock.Match = mockllm.AnthropicRequestMatch{
		MatchType: mockllm.MatchTypeContains,
		Message:   anthropicRequest.Messages[len(anthropicRequest.Messages)-1],
	}
	// Marshal and unmarshal the request to get it in the right format
	reqBytes, err := json.Marshal(anthropicRequest)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytes, &mock.Match)
	require.NoError(t, err)

	config := mockllm.Config{
		Anthropic: []mockllm.AnthropicMock{mock},
	}

	// Start server
	server := mockllm.NewServer(config)
	baseURL, err := server.Start(t.Context())
	require.NoError(t, err)
	defer server.Stop(context.Background()) //nolint:errcheck

	// Make request
	req, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytes))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "test-key")
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	// Check response
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "msg_123", responseBody["id"])
	assert.Equal(t, "message", responseBody["type"])
}

func TestOpenAIModelMatching(t *testing.T) {
	// Create mocks for different models
	gpt4MiniRequest := openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role: "user",
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hello"),
					},
				},
			},
		},
	}

	gpt4Request := openai.ChatCompletionNewParams{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role: "user",
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hello"),
					},
				},
			},
		},
	}

	gpt4MiniResponse := openai.ChatCompletion{
		ID:      "chatcmpl-mini",
		Object:  "chat.completion",
		Created: 1677652288,
		Model:   "gpt-4o-mini",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello from mini!",
				},
				FinishReason: "stop",
			},
		},
	}

	gpt4Response := openai.ChatCompletion{
		ID:      "chatcmpl-gpt4",
		Object:  "chat.completion",
		Created: 1677652288,
		Model:   "gpt-4",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello from GPT-4!",
				},
				FinishReason: "stop",
			},
		},
	}

	// Create mocks with model matching
	var miniMock mockllm.OpenAIMock
	miniMock.Name = "gpt-4o-mini-response"
	miniMock.Response = gpt4MiniResponse
	modelMini := "gpt-4o-mini"
	miniMock.Match = mockllm.OpenAIRequestMatch{
		MatchType: mockllm.MatchTypeExact,
		Message:   gpt4MiniRequest.Messages[len(gpt4MiniRequest.Messages)-1],
		Model:     &modelMini,
	}
	reqBytesMini, err := json.Marshal(gpt4MiniRequest)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytesMini, &miniMock.Match)
	require.NoError(t, err)

	var gpt4Mock mockllm.OpenAIMock
	gpt4Mock.Name = "gpt-4-response"
	gpt4Mock.Response = gpt4Response
	modelGPT4 := "gpt-4"
	gpt4Mock.Match = mockllm.OpenAIRequestMatch{
		MatchType: mockllm.MatchTypeExact,
		Message:   gpt4Request.Messages[len(gpt4Request.Messages)-1],
		Model:     &modelGPT4,
	}
	reqBytesGPT4, err := json.Marshal(gpt4Request)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytesGPT4, &gpt4Mock.Match)
	require.NoError(t, err)

	config := mockllm.Config{
		OpenAI: []mockllm.OpenAIMock{miniMock, gpt4Mock},
	}

	// Start server
	server := mockllm.NewServer(config)
	baseURL, err := server.Start(t.Context())
	require.NoError(t, err)
	defer server.Stop(context.Background()) //nolint:errcheck

	// Test gpt-4o-mini request matches the correct mock
	req, err := http.NewRequest("POST", baseURL+"/v1/chat/completions", bytes.NewReader(reqBytesMini))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-mini", responseBody["id"])

	// Test gpt-4 request matches the correct mock
	req2, err := http.NewRequest("POST", baseURL+"/v1/chat/completions", bytes.NewReader(reqBytesGPT4))
	require.NoError(t, err)
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("Authorization", "Bearer test-key")

	resp2, err := client.Do(req2)
	require.NoError(t, err)
	defer resp2.Body.Close() //nolint:errcheck

	assert.Equal(t, http.StatusOK, resp2.StatusCode)

	var responseBody2 map[string]interface{}
	err = json.NewDecoder(resp2.Body).Decode(&responseBody2)
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-gpt4", responseBody2["id"])
}

func TestAnthropicModelMatching(t *testing.T) {
	// Create requests for different models
	sonnetRequest := anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-20240620",
		MaxTokens: 1000,
		Messages: []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfText: &anthropic.TextBlockParam{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	haikuRequest := anthropic.MessageNewParams{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 1000,
		Messages: []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfText: &anthropic.TextBlockParam{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	sonnetResponse := anthropic.Message{
		ID:   "msg_sonnet",
		Type: "message",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "text",
				Text: "Hello from Sonnet!",
			},
		},
		Model:      "claude-3-5-sonnet-20240620",
		StopReason: "end_turn",
	}

	haikuResponse := anthropic.Message{
		ID:   "msg_haiku",
		Type: "message",
		Role: "assistant",
		Content: []anthropic.ContentBlockUnion{
			{
				Type: "text",
				Text: "Hello from Haiku!",
			},
		},
		Model:      "claude-3-haiku-20240307",
		StopReason: "end_turn",
	}

	// Create mocks with model matching
	var sonnetMock mockllm.AnthropicMock
	sonnetMock.Name = "sonnet-response"
	sonnetMock.Response = sonnetResponse
	modelSonnet := "claude-3-5-sonnet-20240620"
	sonnetMock.Match = mockllm.AnthropicRequestMatch{
		MatchType: mockllm.MatchTypeContains,
		Message:   sonnetRequest.Messages[len(sonnetRequest.Messages)-1],
		Model:     &modelSonnet,
	}
	reqBytesSonnet, err := json.Marshal(sonnetRequest)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytesSonnet, &sonnetMock.Match)
	require.NoError(t, err)

	var haikuMock mockllm.AnthropicMock
	haikuMock.Name = "haiku-response"
	haikuMock.Response = haikuResponse
	modelHaiku := "claude-3-haiku-20240307"
	haikuMock.Match = mockllm.AnthropicRequestMatch{
		MatchType: mockllm.MatchTypeContains,
		Message:   haikuRequest.Messages[len(haikuRequest.Messages)-1],
		Model:     &modelHaiku,
	}
	reqBytesHaiku, err := json.Marshal(haikuRequest)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytesHaiku, &haikuMock.Match)
	require.NoError(t, err)

	config := mockllm.Config{
		Anthropic: []mockllm.AnthropicMock{sonnetMock, haikuMock},
	}

	// Start server
	server := mockllm.NewServer(config)
	baseURL, err := server.Start(t.Context())
	require.NoError(t, err)
	defer server.Stop(context.Background()) //nolint:errcheck

	// Test sonnet request matches the correct mock
	req, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytesSonnet))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "test-key")
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "msg_sonnet", responseBody["id"])

	// Test haiku request matches the correct mock
	req2, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytesHaiku))
	require.NoError(t, err)
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("x-api-key", "test-key")
	req2.Header.Set("anthropic-version", "2023-06-01")

	resp2, err := client.Do(req2)
	require.NoError(t, err)
	defer resp2.Body.Close() //nolint:errcheck

	assert.Equal(t, http.StatusOK, resp2.StatusCode)

	var responseBody2 map[string]interface{}
	err = json.NewDecoder(resp2.Body).Decode(&responseBody2)
	require.NoError(t, err)

	assert.Equal(t, "msg_haiku", responseBody2["id"])
}

func TestHealthCheck(t *testing.T) {
	config := mockllm.Config{}
	server := mockllm.NewServer(config)
	baseURL, err := server.Start(t.Context())
	require.NoError(t, err)
	defer server.Stop(context.Background()) //nolint:errcheck

	resp, err := http.Get(baseURL + "/health")
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "healthy", responseBody["status"])
	assert.Equal(t, "mock-llm", responseBody["service"])
}
