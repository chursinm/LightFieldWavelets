#pragma once
// TODO refine this
class TextureException : public std::exception
{
public:
	virtual char const * what() const { return "Couldn't load texture"; }
};

/*
Fetches an image from disk and transfers it to the GPU as opengl texture.
Supports a wide range of image formats via SDL2_Image.
Throws TextureException if anything goes wrong.

Usage:
-asyncTransferToGPU() until finished() is true

TODO implement using ie. observer
*/
class Texture
{
public:
	/*
	Starts an asynchronous fetch for filename.
	Call asyncTransferToGPU() to create the opengl texture.
	*/
	Texture(const std::string& filename);
	/*
	Starts an asynchronous fetch for filename.
	Call asyncTransferToGPU() to create the opengl texture.
	Will store the texture at glid.
	*/
	Texture(const std::string& filename, GLuint glid);
	~Texture();
	// we do not want a copy ctor
	Texture(const Texture& that) = delete;

	/*
	Transfers the fetched Image to the GPU.
	If the fetch is not ready yet, the function will block for waitFor.
	If the fetch didn't finish in time, this will do nothing.
	*/
	template<class T, class P>
	void asyncTransferToGPU(const std::chrono::duration<T, P>& waitFor)
	{
		if(m_IsFinished) { return; }
		if(!m_SurfaceFuture.valid())
		{
			WARN("No fetch for texture pending");
			throw TextureException();
		}

		auto status = m_SurfaceFuture.wait_for(waitFor);
		if(status == std::future_status::deferred)
		{
			WARN("Invalid future status");
			throw TextureException();
		}
		else if(status == std::future_status::timeout)
		{
			return;
		}
		else if(status == std::future_status::ready)
		{
			transferToGPU(m_SurfaceFuture.get());
		}
	}
	bool finished();
	void bind(unsigned int unit);

private:
	/*
	Starts an asynchronous fetch for the Texture File.
	*/
	void fetchFile();
	void transferToGPU(SDL_Surface* surface);
	std::string m_Filename;
	std::future<SDL_Surface*> m_SurfaceFuture;
	bool m_IsFinished;
	unsigned int m_Width, m_Height;
	GLuint m_GLID;
};

